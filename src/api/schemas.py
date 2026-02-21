"""
Pydantic schemas for all API request / response models.
All schemas enforce strict types to guarantee structured JSON I/O.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


# ─── Shared ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


# ─── Training ────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    project_id: str = Field("TCGA-BRCA", description="TCGA project identifier")
    use_synthetic: bool = Field(True,  description="Use synthetic data instead of real TCGA download")
    use_neural_net: bool = Field(False, description="Use shallow NN instead of ElasticNet")
    n_synthetic_samples: int = Field(400, ge=50, description="Number of synthetic samples (if use_synthetic=True)")


class TrainResponse(BaseModel):
    project_id: str
    baseline_results: Dict[str, Dict[str, float]]
    primary_model_metrics: Dict[str, float]
    gate_passed: bool
    model_artifact_path: str
    stable_biomarkers: List[str]


# ─── Evaluation ──────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    project_id: str = "TCGA-BRCA"
    split: str = Field("test", description="Which split to evaluate: 'val' or 'test'")


class EvalResponse(BaseModel):
    split: str
    auroc: float
    auprc: float
    f1: float
    accuracy: float
    auroc_95ci: Optional[Tuple[float, float]] = None


# ─── Biomarker analysis ──────────────────────────────────────────────────────

class BiomarkerRequest(BaseModel):
    project_id: str = "TCGA-BRCA"


class BiomarkerRecord(BaseModel):
    gene_id: str
    stability_score: float
    evidence_strength_raw: float
    evidence_strength_norm: float
    vulnerability_score: float
    biomarker_confidence_score: float
    high_confidence: bool
    evidence_type: str
    conflicting_findings: bool
    mechanism_summary: Optional[str] = None
    supporting_pmids: Optional[str] = None


class BiomarkerResponse(BaseModel):
    project_id: str
    total_biomarkers: int
    high_confidence_count: int
    biomarkers: List[BiomarkerRecord]


# ─── Literature validation ───────────────────────────────────────────────────

class LiteratureRequest(BaseModel):
    gene: str = Field(..., description="Gene ID (ENSG... or symbol)")
    disease_keyword: str = Field(..., description="Disease search term")
    run_adversarial: bool = Field(True, description="Also run adversarial falsification")


class LiteratureResponse(BaseModel):
    gene: str
    abstracts_found: int
    evidence: Optional[Dict[str, Any]] = None
    adversarial: Optional[Dict[str, Any]] = None


# ─── Cross-cohort ─────────────────────────────────────────────────────────────

class CrossCohortRequest(BaseModel):
    project_id: str = "TCGA-BRCA"
    external_cohort_path: str = Field(..., description="Path to external cohort processed data directory")


class CrossCohortResponse(BaseModel):
    internal_test: Dict[str, Any]
    external_cohort: Dict[str, Any]
    auroc_delta: float
    generalisation_flag: str
    biomarker_overlap_pct: Optional[float] = None


# ─── Biomedical 10-stage audit ───────────────────────────────────────────────

class MultiOmicsFeatures(BaseModel):
    genomics_features: List[str] = Field(default_factory=list)
    transcriptomics_features: List[str] = Field(default_factory=list)
    proteomics_features: List[str] = Field(default_factory=list)


class AuditMetrics(BaseModel):
    auroc: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class BiomedicalAuditRequest(BaseModel):
    project_id: str = Field("TCGA-BRCA", description="Project identifier for result persistence")
    disease_name: str = Field(..., description="Disease context name")
    multi_omics_features: MultiOmicsFeatures
    selected_biomarkers: List[str] = Field(default_factory=list)
    shap_importance_scores: Dict[str, float] = Field(default_factory=dict)
    stability_scores: Dict[str, float] = Field(default_factory=dict)
    model_metrics: AuditMetrics
    training_sample_size: int = Field(..., ge=1)
    validation_sample_size: int = Field(..., ge=1)


class BiomedicalAuditBiomarker(BaseModel):
    name: str
    omics_type: str
    shap_importance: str
    stability_score: str
    stability_classification: str
    biological_evidence_strength: str
    known_pathways: str
    causal_role: str
    counterfactual_direction: str
    therapeutic_status: str
    confidence_level: str


class BiomedicalAuditResponse(BaseModel):
    data_audit: Dict[str, str]
    model_performance_review: Dict[str, str]
    biomarker_analysis: List[BiomedicalAuditBiomarker]
    flagged_unstable_features: List[str]
    high_confidence_targets: List[str]
    overall_system_verdict: str
