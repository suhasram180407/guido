"""
Unit tests for LLM validation components (JSON parsing, schema enforcement).
Uses mocked Groq responses — no real API calls.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pytest
from unittest.mock import patch, MagicMock

from src.llm.groq_validator import _extract_json, _format_abstracts


# ─── _extract_json ─────────────────────────────────────────────────────────

def test_extract_json_clean():
    text = '{"gene": "BRCA1", "evidence_strength": 4}'
    result = _extract_json(text)
    assert result == {"gene": "BRCA1", "evidence_strength": 4}


def test_extract_json_with_markdown_fence():
    text = '```json\n{"gene": "TP53", "evidence_strength": 5}\n```'
    result = _extract_json(text)
    assert result["gene"] == "TP53"


def test_extract_json_invalid_returns_none():
    assert _extract_json("This is not JSON at all") is None


def test_extract_json_embedded_in_text():
    text = 'Some preamble text.\n{"key": "value"}\nSome trailing text.'
    result = _extract_json(text)
    assert result == {"key": "value"}


# ─── _format_abstracts ───────────────────────────────────────────────────────

def test_format_abstracts_empty():
    text = _format_abstracts([])
    assert "No abstracts" in text


def test_format_abstracts_includes_pmid():
    abstracts = [{"pmid": "99999999", "title": "Test study", "abstract": "Test abstract.", "year": "2024"}]
    text = _format_abstracts(abstracts)
    assert "99999999" in text
    assert "Test study" in text


def test_format_abstracts_multiple():
    abstracts = [
        {"pmid": "1", "title": "A", "abstract": "Abstract A", "year": "2020"},
        {"pmid": "2", "title": "B", "abstract": "Abstract B", "year": "2021"},
    ]
    text = _format_abstracts(abstracts)
    assert "[1]" in text and "[2]" in text


# ─── validate_biomarker_evidence (mocked) ────────────────────────────────────

def _mock_groq_response(content: str):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    return mock_client


def test_validate_evidence_valid_json():
    from src.llm.groq_validator import validate_biomarker_evidence
    valid_response = json.dumps({
        "gene": "BRCA1",
        "evidence_strength": 4,
        "evidence_type": "clinical",
        "conflicting_findings": False,
        "conflict_description": None,
        "mechanism_summary": "BRCA1 regulates DNA repair.",
        "supporting_pmids": ["12345678"],
        "reasoning": "Two clinical studies demonstrate upregulation.",
    })
    mock_client = _mock_groq_response(valid_response)

    with patch("src.llm.groq_validator.GROQ_API_KEY", "test_key"):
        result = validate_biomarker_evidence("BRCA1", "breast cancer", [], groq_client=mock_client)

    assert result is not None
    assert result["gene"] == "BRCA1"
    assert result["evidence_strength"] == 4


def test_validate_evidence_invalid_json_retries_and_fails():
    """All retries return invalid JSON → result should be None."""
    from src.llm.groq_validator import validate_biomarker_evidence
    mock_client = _mock_groq_response("NOT VALID JSON AT ALL !!!")

    with patch("src.llm.groq_validator.GROQ_API_KEY", "test_key"):
        result = validate_biomarker_evidence("FAKE", "fake disease", [], groq_client=mock_client)

    assert result is None


def test_adversarial_vulnerability_clamped():
    """Vulnerability score outside [0,1] should be clamped."""
    from src.llm.groq_validator import adversarial_falsify
    response = json.dumps({
        "gene": "TP53",
        "vulnerability_score": 1.5,  # out of range
        "small_sample_studies": False,
        "correlation_only": False,
        "contradictory_evidence": False,
        "confounding_issues": False,
        "cell_line_only": False,
        "weaknesses": [],
        "overall_assessment": "Strong evidence.",
    })
    mock_client = _mock_groq_response(response)

    with patch("src.llm.groq_validator.GROQ_API_KEY", "test_key"):
        result = adversarial_falsify("TP53", "cancer", [], groq_client=mock_client)

    assert result is not None
    assert 0.0 <= result["vulnerability_score"] <= 1.0


def test_biomedical_audit_requires_key():
    from src.llm.groq_validator import run_biomedical_system_audit
    payload = {
        "disease_name": "breast cancer",
        "multi_omics_features": {
            "genomics_features": ["BRCA1"],
            "transcriptomics_features": [],
            "proteomics_features": [],
        },
        "selected_biomarkers": ["BRCA1"],
        "shap_importance_scores": {"BRCA1": 0.9},
        "stability_scores": {"BRCA1": 0.8},
        "model_metrics": {"auroc": 0.8, "accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1": 0.7},
        "training_sample_size": 100,
        "validation_sample_size": 50,
    }
    with patch("src.llm.groq_validator.GROQ_API_KEY", ""):
        with pytest.raises(EnvironmentError):
            run_biomedical_system_audit(payload)


def test_biomedical_audit_strict_success_short_circuit():
    from src.llm.groq_validator import run_biomedical_system_audit
    payload = {
        "disease_name": "breast cancer",
        "multi_omics_features": {
            "genomics_features": ["BRCA1"],
            "transcriptomics_features": ["TP53"],
            "proteomics_features": [],
        },
        "selected_biomarkers": ["BRCA1"],
        "shap_importance_scores": {"BRCA1": 0.9},
        "stability_scores": {"BRCA1": 0.8},
        "model_metrics": {"auroc": 0.8, "accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1": 0.7},
        "training_sample_size": 100,
        "validation_sample_size": 50,
    }
    mock_structured = {
        "data_audit": {
            "feature_sample_ratio": "0.01",
            "overfitting_risk": "LOW",
            "generalization_comment": "ok",
        },
        "model_performance_review": {"performance_quality": "ok", "bias_risk": "low"},
        "biomarker_analysis": [],
        "flagged_unstable_features": [],
        "high_confidence_targets": [],
        "overall_system_verdict": "ok",
    }

    with patch("src.llm.groq_validator.GROQ_API_KEY", "test_key"):
        with patch("src.llm.groq_validator._call_groq_structured", return_value=mock_structured):
            result = run_biomedical_system_audit(payload, groq_client=MagicMock())

    assert result["overall_system_verdict"] == "ok"
