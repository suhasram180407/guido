#!/usr/bin/env python3
"""Comprehensive model comparison with stratified CV, SMOTE, and multiple algorithms.

Tests: LogisticRegression (elastic-net), XGBoost, RandomForest.
Applies SMOTE for class balancing.
Provides stratified CV AUROC, learning curves, and diagnostics.
Selects best model and saves it.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(project: str, n_genes: int = 500, test_size: float = 0.2):
    repo = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(repo))

    from src.api.pipeline import _load_splits
    from config.settings import MODELS_DIR, RESULTS_DIR
    from src.models.metrics import compute_clf_metrics

    logger.info("Loading data splits from cached artifacts...")
    splits = _load_splits(project)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Combine train+val for stratified CV (more data for robust estimates)
    X_fit = pd.concat([X_train, X_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)
    
    logger.info(f"Combined train+val: {X_fit.shape[0]} samples, {X_fit.shape[1]} features")
    logger.info(f"Class distribution: {y_fit.value_counts().to_dict()}")
    
    # Select top n_genes by variance (already done in train, but ensure consistency)
    variances = X_fit.var(axis=0)
    top_genes = variances.nlargest(n_genes).index.tolist()
    
    X_fit_sub = X_fit.loc[:, top_genes]
    X_test_sub = X_test.loc[:, top_genes]
    
    logger.info(f"Selected top {n_genes} genes by variance")
    
    # Define models to compare
    models = {
        "LogisticRegression_ElasticNet": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42, k_neighbors=min(3, y_fit.sum() - 1))),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga", C=1.0, l1_ratio=0.5, max_iter=5000))
        ]),
        "RandomForest": ImbPipeline([
            ("smote", SMOTE(random_state=42, k_neighbors=min(3, y_fit.sum() - 1))),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
        ]),
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = ImbPipeline([
            ("smote", SMOTE(random_state=42, k_neighbors=min(3, y_fit.sum() - 1))),
            ("clf", XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, 
                random_state=42, eval_metric="logloss", use_label_encoder=False, verbosity=0
            ))
        ])
    
    # Stratified K-Fold CV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    results_summary = {}
    best_model_name = None
    best_auroc = -1
    best_model = None
    
    logger.info("Running stratified cross-validation on models...")
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        try:
            cv_results = cross_validate(
                model, X_fit_sub, y_fit, 
                cv=cv, 
                scoring=["roc_auc", "accuracy", "precision", "recall", "f1"],
                n_jobs=1
            )
            
            auroc_mean = cv_results["test_roc_auc"].mean()
            auroc_std = cv_results["test_roc_auc"].std()
            
            results_summary[model_name] = {
                "auroc_mean": float(auroc_mean),
                "auroc_std": float(auroc_std),
                "auroc_scores": cv_results["test_roc_auc"].tolist(),
                "accuracy_mean": float(cv_results["test_accuracy"].mean()),
                "precision_mean": float(cv_results["test_precision"].mean()),
                "recall_mean": float(cv_results["test_recall"].mean()),
                "f1_mean": float(cv_results["test_f1"].mean()),
            }
            
            logger.info(f"  AUROC: {auroc_mean:.4f} (+/- {auroc_std:.4f})")
            logger.info(f"  Accuracy: {cv_results['test_accuracy'].mean():.4f}")
            logger.info(f"  F1: {cv_results['test_f1'].mean():.4f}")
            
            if auroc_mean > best_auroc:
                best_auroc = auroc_mean
                best_model_name = model_name
                best_model = model
        except Exception as e:
            logger.warning(f"  Failed: {e}")
            results_summary[model_name] = {"error": str(e)}
    
    # Train best model on full train+val set and evaluate on test
    if best_model is None:
        logger.error("No model trained successfully!")
        return
    
    logger.info(f"\n=== Best Model: {best_model_name} (AUROC: {best_auroc:.4f}) ===")
    
    best_model.fit(X_fit_sub, y_fit)
    y_prob = best_model.predict_proba(X_test_sub)[:, 1]
    test_metrics = compute_clf_metrics(y_test.values, y_prob)
    
    logger.info(f"Test AUROC: {test_metrics.get('auroc', 'N/A')}")
    logger.info(f"Test Accuracy: {test_metrics.get('accuracy', 'N/A')}")
    logger.info(f"Test F1: {test_metrics.get('f1', 'N/A')}")
    
    # Save best model
    mdl_dir = Path(MODELS_DIR) / project
    mdl_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, mdl_dir / "primary_model_improved.joblib")
    
    # Save results
    out_dir = Path(RESULTS_DIR) / project
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary["best_model"] = best_model_name
    results_summary["best_cv_auroc"] = float(best_auroc)
    results_summary["test_metrics"] = test_metrics
    results_summary["n_train_samples"] = int(X_fit.shape[0])
    results_summary["n_test_samples"] = int(X_test.shape[0])
    results_summary["n_features"] = int(X_fit_sub.shape[1])
    
    (out_dir / "model_comparison_results.json").write_text(
        json.dumps(results_summary, indent=2)
    )
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(json.dumps(results_summary, indent=2))
    print("="*60)
    print(f"Best model saved to: {mdl_dir / 'primary_model_improved.joblib'}")
    print(f"Results saved to: {out_dir / 'model_comparison_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--n-genes", type=int, default=500)
    args = parser.parse_args()
    main(args.project, args.n_genes)
