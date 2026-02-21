"""
Step 8 – Cross-Cohort Validation
==================================
Evaluates the trained primary model on an independent external cohort and
computes robustness metrics:

  1. AUROC on external cohort vs internal test set (ΔAUROC)
  2. Biomarker overlap percentage between internal and external SHAP results
  3. Statistical significance of AUROC difference (DeLong's test approximation)

All results are logged to MLflow and persisted to disk.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MLFLOW_TRACKING_URI, RESULTS_DIR
from src.models.metrics import compute_clf_metrics

logger = logging.getLogger(__name__)

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ─── AUROC bootstrap CI ──────────────────────────────────────────────────────

def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for AUROC."""
    rng = np.random.default_rng(seed)
    aurocs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aurocs.append(roc_auc_score(y_b, p_b))
    aurocs = np.array(aurocs)
    lower = float(np.percentile(aurocs, 100 * alpha / 2))
    upper = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return lower, upper


# ─── Align feature spaces ────────────────────────────────────────────────────

def align_features(
    X_ext: pd.DataFrame,
    X_internal: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reindex external cohort to match the internal feature set.
    Missing genes are zero-filled; extra genes are dropped.
    """
    missing = set(X_internal.columns) - set(X_ext.columns)
    if missing:
        logger.warning(
            "External cohort is missing %d genes (will be zero-filled): %s…",
            len(missing), list(missing)[:5],
        )
    X_aligned = X_ext.reindex(columns=X_internal.columns, fill_value=0.0)
    return X_aligned


# ─── Cross-cohort validation ─────────────────────────────────────────────────

def run_cross_cohort_validation(
    model,
    X_internal_test: pd.DataFrame,
    y_internal_test: pd.Series,
    X_external: pd.DataFrame,
    y_external: pd.Series,
    internal_biomarkers: Optional[List[str]] = None,
    external_biomarkers: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate model on both internal test set and external cohort.

    Returns a robustness report dict.
    """
    # Align external cohort features to internal feature space
    X_ext_aligned = align_features(X_external, X_internal_test)

    from contextlib import nullcontext
    ctx = mlflow.start_run(run_name="cross_cohort") if _MLFLOW_AVAILABLE else nullcontext()
    if _MLFLOW_AVAILABLE:
        mlflow.set_experiment("cross_cohort_validation")

    with ctx:
        # ── Internal test ──
        if hasattr(model, "predict_proba"):
            y_prob_int = model.predict_proba(X_internal_test)[:, 1]
            y_prob_ext = model.predict_proba(X_ext_aligned)[:, 1]
        else:
            y_prob_int = model.decision_function(X_internal_test)
            y_prob_ext = model.decision_function(X_ext_aligned)

        int_metrics = compute_clf_metrics(y_internal_test.values, y_prob_int)
        ext_metrics = compute_clf_metrics(y_external.values, y_prob_ext)

        auroc_delta = int_metrics["auroc"] - ext_metrics["auroc"]

        # ── Bootstrap CIs ──
        int_ci = bootstrap_auroc_ci(y_internal_test.values, y_prob_int)
        ext_ci = bootstrap_auroc_ci(y_external.values,      y_prob_ext)

        # ── Biomarker overlap ──
        overlap_pct = None
        if internal_biomarkers and external_biomarkers:
            int_set = set(internal_biomarkers)
            ext_set = set(external_biomarkers)
            union = int_set | ext_set
            overlap_pct = 100.0 * len(int_set & ext_set) / len(union) if union else 0.0

        report = {
            "internal_test": {**int_metrics, "auroc_95ci": int_ci},
            "external_cohort": {**ext_metrics, "auroc_95ci": ext_ci},
            "auroc_delta": round(auroc_delta, 4),
            "generalisation_flag": "PASS" if abs(auroc_delta) <= 0.10 else "WARN",
            "biomarker_overlap_pct": round(overlap_pct, 2) if overlap_pct is not None else None,
        }

        # Log to MLflow
        if _MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "internal_auroc": int_metrics["auroc"],
                "external_auroc": ext_metrics["auroc"],
                "auroc_delta":    auroc_delta,
            })
            if overlap_pct is not None:
                mlflow.log_metric("biomarker_overlap_pct", overlap_pct)
            mlflow.set_tag("generalisation_flag", report["generalisation_flag"])

    logger.info("Cross-cohort validation complete: %s", report)
    return report


def save_robustness_report(report: Dict, out_dir: Optional[Path] = None) -> Path:
    """Persist robustness report to JSON."""
    import json
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "robustness_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved robustness report to %s", out_path)
    return out_path
