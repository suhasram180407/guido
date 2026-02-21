"""
Step 2 – Baseline Models
========================
Logistic Regression (L1), Random Forest, XGBoost.
5-fold stratified cross-validation.
All metrics logged to MLflow.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CV_FOLDS, MLFLOW_TRACKING_URI, RANDOM_SEED
from src.models.metrics import compute_clf_metrics, cv_aggregate_metrics, log_metrics_to_mlflow

logger = logging.getLogger(__name__)

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ─── Model factories ─────────────────────────────────────────────────────────

def build_lr_l1(C: float = 0.1) -> Pipeline:
    """Logistic Regression with L1 regularisation inside a scaling pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            l1_ratio=1.0,   # pure L1 (replaces deprecated penalty='l1')
            C=C,
            max_iter=2000,
            random_state=RANDOM_SEED,
        )),
    ])


def build_random_forest(n_estimators: int = 200, max_depth: Optional[int] = 6) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_xgboost(n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.05):
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


# ─── Cross-validation runner ─────────────────────────────────────────────────

def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    experiment_name: str = "baseline_models",
    hyperparams: Optional[Dict] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Run stratified k-fold CV, log each fold and aggregate metrics to MLflow.

    Returns
    -------
    agg_metrics  : aggregated mean/std metrics dict
    fold_metrics : list of per-fold metric dicts
    """
    mlflow.set_experiment(experiment_name)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_metrics: List[Dict[str, float]] = []
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    def _start_run(name):
        if _MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
            return mlflow.start_run(run_name=f"{name}_cv")
        from contextlib import nullcontext
        return nullcontext()

    with _start_run(model_name):
        if _MLFLOW_AVAILABLE:
            mlflow.set_tag("model_name", model_name)
            if hyperparams:
                mlflow.log_params(hyperparams)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            import copy
            fold_model = copy.deepcopy(model)
            fold_model.fit(X_tr, y_tr)

            # Get probability scores
            if hasattr(fold_model, "predict_proba"):
                y_prob = fold_model.predict_proba(X_val)[:, 1]
            else:
                y_prob = fold_model.decision_function(X_val)

            fold_m = compute_clf_metrics(y_val, y_prob)
            fold_m["fold"] = fold_idx
            fold_metrics.append(fold_m)

            log_metrics_to_mlflow(
                {k: v for k, v in fold_m.items() if k != "fold"},
                step=fold_idx,
            )
            logger.info("  Fold %d/%d  AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                        fold_idx + 1, CV_FOLDS,
                        fold_m["auroc"], fold_m["auprc"], fold_m["f1"])

        agg = cv_aggregate_metrics(fold_metrics)
        log_metrics_to_mlflow(agg, tags={"phase": "cv_aggregate"})
        logger.info("%s CV results: %s", model_name, agg)

    return agg, fold_metrics


# ─── Run all baselines ────────────────────────────────────────────────────────

def run_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Train all baseline models with CV. Returns dict of {model_name: agg_metrics}.
    """
    baselines = {
        "LR_L1": (build_lr_l1(), {"C": 0.1, "penalty": "l1"}),
        "RF_200": (build_random_forest(), {"n_estimators": 200, "max_depth": 6}),
    }
    if _XGB_AVAILABLE:
        baselines["XGB_200"] = (build_xgboost(), {"n_estimators": 200, "max_depth": 4, "lr": 0.05})
    else:
        logger.warning("xgboost not installed — skipping XGBoost baseline")

    results: Dict[str, Dict[str, float]] = {}
    for name, (model, params) in baselines.items():
        logger.info("Running CV for %s …", name)
        agg, _ = cross_validate_model(model, X_train, y_train, model_name=name, hyperparams=params)
        results[name] = agg
    return results
