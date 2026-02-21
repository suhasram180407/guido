"""
Step 7 – Biomarker Confidence Score
=====================================
Computes the final BCS for each stable biomarker:

    BCS = stability_score × normalised_evidence_strength × (1 − vulnerability_score)

Where:
  - stability_score         ∈ [0,1]  (fold appearance frequency, from SHAP)
  - normalised_evidence     ∈ [0,1]  (raw LLM score 1–5 → /5)
  - vulnerability_score     ∈ [0,1]  (adversarial LLM, higher = weaker)

A biomarker with BCS ≥ BCS_THRESHOLD is promoted to "high-confidence".
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RESULTS_DIR

logger = logging.getLogger(__name__)

BCS_THRESHOLD = 0.30   # minimum to be considered high-confidence


def compute_biomarker_confidence(
    stable_biomarkers: List[str],
    stability_scores: Dict[str, float],
    llm_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compute Biomarker Confidence Score for each stable biomarker.

    Parameters
    ----------
    stable_biomarkers : ordered list of gene IDs
    stability_scores  : {gene: stability ∈ [0,1]}
    llm_results       : {gene: {"evidence": {...}, "adversarial": {...}}}

    Returns
    -------
    DataFrame with columns:
        gene_id, stability_score, evidence_strength_raw,
        evidence_strength_norm, vulnerability_score,
        biomarker_confidence_score, high_confidence,
        evidence_type, conflicting_findings, mechanism_summary
    """
    rows = []
    for gene in stable_biomarkers:
        stab = stability_scores.get(gene, 0.0)

        ev_raw   = 1.0  # default: neutral weak evidence
        vuln     = 0.5  # default: moderate uncertainty
        ev_type  = "unclear"
        conflict = False
        mechanism = None
        evidence_pmids: List[str] = []

        gene_llm = llm_results.get(gene, {})

        evidence_dict = gene_llm.get("evidence")
        if evidence_dict:
            ev_raw  = float(evidence_dict.get("evidence_strength", 1))
            ev_type = evidence_dict.get("evidence_type", "unclear")
            conflict = bool(evidence_dict.get("conflicting_findings", False))
            mechanism = evidence_dict.get("mechanism_summary")
            evidence_pmids = evidence_dict.get("supporting_pmids", [])

        adv_dict = gene_llm.get("adversarial")
        if adv_dict:
            vuln = float(adv_dict.get("vulnerability_score", 0.5))
            vuln = max(0.0, min(1.0, vuln))

        ev_norm = (ev_raw - 1.0) / 4.0   # map [1,5] → [0,1]
        bcs = stab * ev_norm * (1.0 - vuln)

        rows.append({
            "gene_id":                     gene,
            "stability_score":             round(stab, 4),
            "evidence_strength_raw":       ev_raw,
            "evidence_strength_norm":      round(ev_norm, 4),
            "vulnerability_score":         round(vuln, 4),
            "biomarker_confidence_score":  round(bcs, 4),
            "high_confidence":             bcs >= BCS_THRESHOLD,
            "evidence_type":               ev_type,
            "conflicting_findings":        conflict,
            "mechanism_summary":           mechanism,
            "supporting_pmids":            ",".join(str(p) for p in evidence_pmids),
        })

    df = pd.DataFrame(rows).sort_values("biomarker_confidence_score", ascending=False)
    df = df.reset_index(drop=True)
    return df


def save_confidence_scores(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Path:
    """Persist BCS table to CSV and return the path."""
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "biomarker_confidence_scores.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved biomarker confidence scores (%d rows) to %s", len(df), out_path)
    return out_path


def high_confidence_biomarkers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only high-confidence biomarkers."""
    return df[df["high_confidence"]].copy()
