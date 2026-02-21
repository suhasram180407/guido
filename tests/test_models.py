"""
Unit tests for ML models and metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.models.metrics import compute_clf_metrics, cv_aggregate_metrics
from src.models.baselines import build_lr_l1, build_random_forest
from src.models.primary_model import build_elasticnet, passes_gate
from src.models.confidence_scoring import compute_biomarker_confidence

try:
    import xgboost as _xgboost_check  # noqa: F401
    from src.models.baselines import build_xgboost
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ─── Metrics ─────────────────────────────────────────────────────────────────

def test_compute_clf_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_clf_metrics(y_true, y_prob)
    assert m["auroc"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_compute_clf_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.3, 0.7, 0.4, 0.6])
    m = compute_clf_metrics(y_true, y_prob)
    assert {"auroc", "auprc", "f1", "accuracy"} == set(m.keys())


def test_cv_aggregate_metrics():
    folds = [{"auroc": 0.8, "auprc": 0.75}, {"auroc": 0.85, "auprc": 0.80}]
    agg = cv_aggregate_metrics(folds)
    assert abs(agg["auroc_mean"] - 0.825) < 1e-6
    assert "auroc_std" in agg


# ─── Baseline models ─────────────────────────────────────────────────────────

def _make_xy(n=100, p=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, p)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(int)
    return X, y


def test_lr_l1_fit_predict():
    X, y = _make_xy()
    model = build_lr_l1()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (100,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_random_forest_fit_predict():
    X, y = _make_xy()
    model = build_random_forest(n_estimators=10)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (100,)


def test_xgboost_fit_predict():
    if not _HAS_XGB:
        pytest.skip("xgboost not installed")
    X, y = _make_xy()
    model = build_xgboost(n_estimators=10)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (100,)


# ─── Primary model ───────────────────────────────────────────────────────────

def test_elasticnet_fit_predict():
    X = pd.DataFrame(np.random.rand(80, 30))
    y = pd.Series([0] * 40 + [1] * 40)
    model = build_elasticnet()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (80,)


def test_passes_gate_true():
    baseline = {"LR": {"auroc_mean": 0.70}, "RF": {"auroc_mean": 0.72}}
    assert passes_gate(0.72, baseline) is True


def test_passes_gate_false():
    baseline = {"LR": {"auroc_mean": 0.80}, "RF": {"auroc_mean": 0.85}}
    assert passes_gate(0.70, baseline) is False


# ─── Confidence scoring ────────────────────────────────────────────────────────

def test_confidence_scoring_no_llm():
    """BCS without LLM results should fall back to defaults."""
    genes = ["GENE1", "GENE2", "GENE3"]
    stability = {"GENE1": 0.8, "GENE2": 0.6, "GENE3": 1.0}
    llm = {}  # empty — simulate no LLM data

    df = compute_biomarker_confidence(genes, stability, llm)
    assert len(df) == 3
    assert "biomarker_confidence_score" in df.columns
    assert (df["biomarker_confidence_score"] >= 0).all()


def test_confidence_scoring_with_llm():
    genes = ["BRCA1"]
    stability = {"BRCA1": 1.0}
    llm = {
        "BRCA1": {
            "evidence": {
                "evidence_strength": 5,
                "evidence_type": "clinical",
                "conflicting_findings": False,
                "mechanism_summary": "Drives homologous recombination repair.",
                "supporting_pmids": ["12345678"],
            },
            "adversarial": {"vulnerability_score": 0.1},
        }
    }
    df = compute_biomarker_confidence(genes, stability, llm)
    row = df.iloc[0]
    assert row["evidence_strength_raw"] == 5
    assert row["vulnerability_score"] == 0.1
    # BCS = 1.0 × (4/4) × (1-0.1) = 0.9
    assert abs(row["biomarker_confidence_score"] - 0.9) < 1e-3
    assert bool(row["high_confidence"]) is True
