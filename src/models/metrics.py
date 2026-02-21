"""
Shared metrics and MLflow logging utilities for all models.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def compute_clf_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute AUROC, AUPRC, F1, and Accuracy.

    Parameters
    ----------
    y_true        : true binary labels
    y_pred_proba  : predicted probabilities for positive class
    threshold     : classification threshold (default 0.5)
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "auroc":    float(roc_auc_score(y_true, y_pred_proba)),
        "auprc":    float(average_precision_score(y_true, y_pred_proba)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def log_metrics_to_mlflow(
    metrics: Dict[str, float],
    params: Optional[Dict] = None,
    tags: Optional[Dict] = None,
    step: Optional[int] = None,
) -> None:
    """Log metrics (and optionally params/tags) to the active MLflow run."""
    try:
        import mlflow
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)
    except ImportError:
        pass  # mlflow optional for local development


def cv_aggregate_metrics(fold_metrics: list[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate per-fold metric dicts into mean ± std summary.

    Returns dict with keys  "<metric>_mean" and "<metric>_std".
    """
    keys = fold_metrics[0].keys()
    agg: Dict[str, float] = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"]  = float(vals.std())
    return agg
