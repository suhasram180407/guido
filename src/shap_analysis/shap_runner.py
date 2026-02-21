"""
Step 4 – SHAP Biomarker Analysis
==================================
Computes SHAP values on validation folds (using the primary model or best
baseline), identifies top-N gene features per fold, and aggregates across
folds to extract stable biomarkers (present in >= MIN_FOLD_APPEARANCES folds).

The module is model-agnostic: it uses TreeExplainer for tree-based models
and KernelExplainer (with background sampling) for others.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    CV_FOLDS,
    MIN_FOLD_APPEARANCES,
    RANDOM_SEED,
    RESULTS_DIR,
    TOP_GENES_PER_FOLD,
)

logger = logging.getLogger(__name__)


# ─── Explainer factory ───────────────────────────────────────────────────────

def get_explainer(model, X_background: np.ndarray) -> shap.Explainer:
    """
    Return an appropriate SHAP explainer based on model type.
    Uses TreeExplainer for sklearn tree ensembles and XGBoost;
    LinearExplainer for linear models; KernelExplainer as fallback.
    """
    model_cls = type(model).__name__.lower()
    # Handle sklearn Pipeline: inspect inner estimator
    inner = model
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("clf", model)
    inner_cls = type(inner).__name__.lower()

    if any(k in inner_cls for k in ("forest", "gradient", "xgb", "lgbm", "tree")):
        logger.debug("Using TreeExplainer")
        return shap.TreeExplainer(inner)

    if any(k in inner_cls for k in ("logistic", "linear", "ridge", "lasso", "elasticnet")):
        logger.debug("Using LinearExplainer")
        return shap.LinearExplainer(inner, X_background)

    # Generic fallback – subsample background for speed
    bg_sample = shap.sample(X_background, min(100, len(X_background)))
    logger.debug("Using KernelExplainer (slow — consider tree-based model)")
    return shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:, 1], bg_sample
    )


# ─── Core SHAP computation ────────────────────────────────────────────────────

def compute_shap_fold(
    model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: List[str],
    top_n: int = TOP_GENES_PER_FOLD,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute SHAP values for validation samples.

    Returns
    -------
    shap_values     : (n_val_samples × n_features) absolute mean SHAP per gene
    top_genes       : list of top_n gene names by mean |SHAP|
    """
    explainer = get_explainer(model, X_train)

    # Handle Pipeline: transform X_val through all steps except last
    X_val_transformed = X_val
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        for step in steps[:-1]:  # all but final estimator
            X_val_transformed = step.transform(X_val_transformed)

    sv = explainer.shap_values(X_val_transformed)

    # For binary classifiers some explainers return a list [neg, pos]
    if isinstance(sv, list):
        sv = sv[1]

    mean_abs_shap = np.abs(sv).mean(axis=0)  # (n_features,)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_genes = [feature_names[i] for i in top_idx]

    return mean_abs_shap, top_genes


# ─── Cross-validated SHAP aggregation ────────────────────────────────────────

def shap_cross_validate(
    model_builder,           # callable() -> untrained model
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = TOP_GENES_PER_FOLD,
    min_folds: int = MIN_FOLD_APPEARANCES,
) -> Tuple[List[str], Dict[str, float], Dict[str, int]]:
    """
    Run stratified k-fold CV, compute SHAP per fold, aggregate stable biomarkers.

    Returns
    -------
    stable_biomarkers  : genes appearing in >= min_folds folds (sorted by frequency)
    stability_scores   : {gene: freq/CV_FOLDS} normalised stability [0,1]
    fold_counts        : {gene: count}
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    feature_names = list(X.columns)
    X_arr = X.values
    y_arr = y.values

    gene_counter: Counter = Counter()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr = y_arr[train_idx]

        import copy
        model = copy.deepcopy(model_builder())
        model.fit(X_tr, y_tr)

        _, top_genes = compute_shap_fold(model, X_tr, X_val, feature_names, top_n)
        gene_counter.update(top_genes)
        logger.info("Fold %d/%d: top gene = %s", fold_idx + 1, CV_FOLDS, top_genes[0])

    fold_counts = dict(gene_counter)
    stable_biomarkers = [
        gene for gene, cnt in sorted(gene_counter.items(), key=lambda x: -x[1])
        if cnt >= min_folds
    ]
    stability_scores = {gene: fold_counts[gene] / CV_FOLDS for gene in stable_biomarkers}

    logger.info(
        "Stable biomarkers: %d (from %d candidates, threshold ≥%d folds)",
        len(stable_biomarkers), len(fold_counts), min_folds,
    )
    return stable_biomarkers, stability_scores, fold_counts


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_biomarker_results(
    stable_biomarkers: List[str],
    stability_scores: Dict[str, float],
    fold_counts: Dict[str, int],
    out_dir: Optional[Path] = None,
) -> Path:
    """Persist biomarker table to CSV."""
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "gene_id": stable_biomarkers,
        "stability_score": [stability_scores[g] for g in stable_biomarkers],
        "fold_count": [fold_counts[g] for g in stable_biomarkers],
    })
    out_path = out_dir / "stable_biomarkers.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d stable biomarkers to %s", len(df), out_path)
    return out_path
