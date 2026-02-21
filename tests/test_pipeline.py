"""
Pipeline-level tests for orchestration helpers.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_run_biomedical_audit_pipeline_persists_report(tmp_path):
    from src.api.pipeline import run_biomedical_audit_pipeline

    models_dir = tmp_path / "models_artifacts"
    results_dir = tmp_path / "results"

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

    mock_report = {
        "data_audit": {
            "feature_sample_ratio": "1.5",
            "overfitting_risk": "MODERATE or LOW",
            "generalization_comment": "External validation recommended.",
        },
        "model_performance_review": {
            "performance_quality": "Good",
            "bias_risk": "Moderate",
        },
        "biomarker_analysis": [],
        "flagged_unstable_features": ["TP53"],
        "high_confidence_targets": ["BRCA1"],
        "overall_system_verdict": "Scientifically plausible with caveats.",
    }

    with patch("src.api.pipeline._get_settings", return_value=(models_dir, results_dir, "TCGA-BRCA", 42)):
        with patch("src.llm.groq_validator.run_biomedical_system_audit", return_value=mock_report):
            result = run_biomedical_audit_pipeline(payload)

    assert result["overall_system_verdict"] == mock_report["overall_system_verdict"]

    out_path = results_dir / "TCGA-BRCA" / "biomedical_audit_breast_cancer.json"
    assert out_path.exists()
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert persisted == mock_report
