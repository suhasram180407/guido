"""
Integration tests for FastAPI endpoints using HTTPX test client.
All heavy pipeline operations are mocked to keep tests fast and offline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import app (no heavy side effects at import time)
from src.api.app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_train_synthetic():
    """Train with synthetic data — no TCGA download required."""
    mock_result = {
        "project_id": "TCGA-BRCA",
        "baseline_results": {
            "LR_L1": {"auroc_mean": 0.75, "auroc_std": 0.02},
        },
        "primary_model_metrics": {"auroc": 0.78, "auprc": 0.70, "f1": 0.65, "accuracy": 0.72},
        "gate_passed": True,
        "model_artifact_path": "/app/models/primary.joblib",
        "stable_biomarkers": ["ENSG00000000001", "ENSG00000000002"],
    }
    with patch("src.api.pipeline.run_training_pipeline", return_value=mock_result):
        resp = client.post("/train", json={
            "project_id": "TCGA-BRCA",
            "use_synthetic": True,
            "use_neural_net": False,
            "n_synthetic_samples": 200,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["gate_passed"] is True
    assert len(data["stable_biomarkers"]) > 0


def test_evaluate_invalid_split():
    resp = client.post("/evaluate", json={"project_id": "TCGA-BRCA", "split": "invalid"})
    assert resp.status_code == 400


def test_evaluate_valid():
    mock_result = {
        "auroc": 0.76, "auprc": 0.68, "f1": 0.64, "accuracy": 0.72,
        "auroc_95ci": [0.71, 0.81], "split": "test",
    }
    with patch("src.api.pipeline.run_evaluation_pipeline", return_value=mock_result):
        resp = client.post("/evaluate", json={"project_id": "TCGA-BRCA", "split": "test"})
    assert resp.status_code == 200
    assert resp.json()["auroc"] == 0.76


def test_literature_no_groq_key():
    """Without GROQ_API_KEY the endpoint should return 503."""
    with patch.dict("os.environ", {}, clear=True):
        # Ensure no GROQ key available
        import os
        os.environ.pop("GROQ_API_KEY", None)
        resp = client.post("/literature", json={
            "gene": "BRCA1",
            "disease_keyword": "breast cancer",
            "run_adversarial": False,
        })
    assert resp.status_code == 503


def test_biomarkers_no_groq_key():
    import os
    os.environ.pop("GROQ_API_KEY", None)
    resp = client.post("/biomarkers", json={"project_id": "TCGA-BRCA"})
    assert resp.status_code == 503


def test_audit_no_groq_key():
    import os
    os.environ.pop("GROQ_API_KEY", None)
    resp = client.post("/audit", json={
        "project_id": "TCGA-BRCA",
        "disease_name": "breast cancer",
        "multi_omics_features": {
            "genomics_features": ["BRCA1"],
            "transcriptomics_features": ["TP53"],
            "proteomics_features": [],
        },
        "selected_biomarkers": ["BRCA1", "TP53"],
        "shap_importance_scores": {"BRCA1": 0.9, "TP53": 0.7},
        "stability_scores": {"BRCA1": 0.8, "TP53": 0.6},
        "model_metrics": {
            "auroc": 0.82,
            "accuracy": 0.76,
            "precision": 0.75,
            "recall": 0.72,
            "f1": 0.73,
        },
        "training_sample_size": 200,
        "validation_sample_size": 100,
    })
    assert resp.status_code == 503


def test_audit_valid_mocked():
    mock_result = {
        "data_audit": {
            "feature_sample_ratio": "2.0",
            "overfitting_risk": "MODERATE or LOW",
            "generalization_comment": "Model appears reasonable; external validation is still required.",
        },
        "model_performance_review": {
            "performance_quality": "Good",
            "bias_risk": "Moderate",
        },
        "biomarker_analysis": [
            {
                "name": "BRCA1",
                "omics_type": "genomics",
                "shap_importance": "0.9",
                "stability_score": "0.8",
                "stability_classification": "Highly Stable",
                "biological_evidence_strength": "Strong",
                "known_pathways": "DNA repair",
                "causal_role": "Upstream driver",
                "counterfactual_direction": "Risk likely decreases if expression decreases",
                "therapeutic_status": "Clinical trial candidate",
                "confidence_level": "High Confidence Biomarker",
            }
        ],
        "flagged_unstable_features": [],
        "high_confidence_targets": ["BRCA1"],
        "overall_system_verdict": "Scientifically plausible with moderate clinical readiness.",
    }
    payload = {
        "project_id": "TCGA-BRCA",
        "disease_name": "breast cancer",
        "multi_omics_features": {
            "genomics_features": ["BRCA1"],
            "transcriptomics_features": ["TP53"],
            "proteomics_features": [],
        },
        "selected_biomarkers": ["BRCA1", "TP53"],
        "shap_importance_scores": {"BRCA1": 0.9, "TP53": 0.7},
        "stability_scores": {"BRCA1": 0.8, "TP53": 0.6},
        "model_metrics": {
            "auroc": 0.82,
            "accuracy": 0.76,
            "precision": 0.75,
            "recall": 0.72,
            "f1": 0.73,
        },
        "training_sample_size": 200,
        "validation_sample_size": 100,
    }
    with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
        with patch("src.api.pipeline.run_biomedical_audit_pipeline", return_value=mock_result):
            resp = client.post("/audit", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["data_audit"]["overfitting_risk"] == "MODERATE or LOW"
    assert data["high_confidence_targets"] == ["BRCA1"]
