"""
Step 5 (Part 2) + Step 6 – Groq LLM Evidence Validation & Adversarial Mode
============================================================================
All LLM prompts operate strictly in grounded-reasoning mode:
  - The model receives ONLY the provided PubMed abstracts as evidence.
  - It is explicitly forbidden from relying on prior training knowledge.
  - All outputs are JSON-only and are validated before acceptance.
  - Invalid JSON responses are retried up to MAX_RETRIES times then rejected.

Two main functions:
  validate_biomarker_evidence()  — Step 5 scoring
  adversarial_falsify()          — Step 6 vulnerability extraction
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    GROQ_AUDIT_MODEL,
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

BIOMED_AUDIT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "data_audit": {
            "type": "object",
            "properties": {
                "feature_sample_ratio": {"type": "string"},
                "overfitting_risk": {"type": "string"},
                "generalization_comment": {"type": "string"},
            },
            "required": ["feature_sample_ratio", "overfitting_risk", "generalization_comment"],
            "additionalProperties": False,
        },
        "model_performance_review": {
            "type": "object",
            "properties": {
                "performance_quality": {"type": "string"},
                "bias_risk": {"type": "string"},
            },
            "required": ["performance_quality", "bias_risk"],
            "additionalProperties": False,
        },
        "biomarker_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "omics_type": {"type": "string"},
                    "shap_importance": {"type": "string"},
                    "stability_score": {"type": "string"},
                    "stability_classification": {"type": "string"},
                    "biological_evidence_strength": {"type": "string"},
                    "known_pathways": {"type": "string"},
                    "causal_role": {"type": "string"},
                    "counterfactual_direction": {"type": "string"},
                    "therapeutic_status": {"type": "string"},
                    "confidence_level": {"type": "string"},
                },
                "required": [
                    "name",
                    "omics_type",
                    "shap_importance",
                    "stability_score",
                    "stability_classification",
                    "biological_evidence_strength",
                    "known_pathways",
                    "causal_role",
                    "counterfactual_direction",
                    "therapeutic_status",
                    "confidence_level",
                ],
                "additionalProperties": False,
            },
        },
        "flagged_unstable_features": {"type": "array", "items": {"type": "string"}},
        "high_confidence_targets": {"type": "array", "items": {"type": "string"}},
        "overall_system_verdict": {"type": "string"},
    },
    "required": [
        "data_audit",
        "model_performance_review",
        "biomarker_analysis",
        "flagged_unstable_features",
        "high_confidence_targets",
        "overall_system_verdict",
    ],
    "additionalProperties": False,
}

# ─── Prompts ─────────────────────────────────────────────────────────────────

EVIDENCE_SYSTEM_PROMPT = """You are a biomedical literature analyst. Your task is to evaluate
scientific evidence from the provided PubMed abstracts only.

CRITICAL RULES:
1. You must ONLY use information explicitly stated in the provided abstracts.
2. Do NOT use your training knowledge or infer beyond what the text says.
3. You must respond with VALID JSON only — no prose, no code fences, no explanation.
4. If the abstracts provide no relevant information, reflect that in low scores.

Output JSON schema (strictly follow this):
{
  "gene": "<gene_id>",
  "evidence_strength": <integer 1-5>,
  "evidence_type": "<clinical|animal|in_vitro|mixed|unclear>",
  "conflicting_findings": <true|false>,
  "conflict_description": "<string or null>",
  "mechanism_summary": "<one sentence derived ONLY from provided text, or null>",
  "supporting_pmids": [<list of PMID strings>],
  "reasoning": "<brief justification citing abstract content>"
}

Evidence strength scale:
  5 = Multiple independent clinical studies with large cohorts
  4 = Single strong clinical study or multiple animal/mechanistic studies
  3 = Single animal study or multiple in-vitro studies with functional data
  2 = In-vitro data only or correlational clinical data
  1 = Only weak associations or conflicting evidence
"""

ADVERSARIAL_SYSTEM_PROMPT = """You are a scientific peer reviewer performing adversarial analysis.
Your goal is to identify weaknesses in the evidence presented.

CRITICAL RULES:
1. Base your critique EXCLUSIVELY on the provided abstracts.
2. Respond with VALID JSON only — no prose, no code fences.
3. Do NOT fabricate weaknesses not supported by the text.

Output JSON schema:
{
  "gene": "<gene_id>",
  "vulnerability_score": <float 0.0-1.0>,
  "small_sample_studies": <true|false>,
  "correlation_only": <true|false>,
  "contradictory_evidence": <true|false>,
  "confounding_issues": <true|false>,
  "cell_line_only": <true|false>,
  "weaknesses": ["<weakness 1 citing text>", "..."],
  "overall_assessment": "<brief critique grounded in provided text>"
}

Vulnerability score guide:
  0.0-0.2 = Strong consistent evidence, minimal weaknesses
  0.2-0.4 = Minor methodological concerns
  0.4-0.6 = Significant limitations (small N, in-vitro only, etc.)
  0.6-0.8 = Major weaknesses (conflicting results, confounding)
  0.8-1.0 = Evidence is critically flawed or nearly absent
"""


# ─── Utilities ───────────────────────────────────────────────────────────────

def _format_abstracts(abstracts: List[Dict[str, str]]) -> str:
    """Concatenate abstracts into a structured text block for the prompt."""
    if not abstracts:
        return "No abstracts available."
    parts = []
    for i, ab in enumerate(abstracts, 1):
        parts.append(
            f"[{i}] PMID:{ab.get('pmid','N/A')} ({ab.get('year','?')})\n"
            f"Title: {ab.get('title','')}\n"
            f"Abstract: {ab.get('abstract','(no abstract)')}\n"
        )
    return "\n".join(parts)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the first JSON object from a response string.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# Hardcode the model to avoid stale config imports
_ACTIVE_MODEL = "llama-3.3-70b-versatile"


def _call_groq(client: Groq, system: str, user: str) -> Optional[Dict[str, Any]]:
    """
    Call Groq API with retry on invalid JSON.
    Returns parsed dict or None if all retries fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=_ACTIVE_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=GROQ_TEMPERATURE,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
            logger.warning("Attempt %d: invalid JSON response, retrying…", attempt)
        except Exception as exc:
            logger.warning("Attempt %d: Groq API error: %s", attempt, exc)

    logger.error("All %d attempts failed — rejecting output", MAX_RETRIES)
    return None


def _call_groq_structured(
    client: Groq,
    *,
    system: str,
    user: str,
    schema_name: str,
    schema: Dict[str, Any],
    model: str,
) -> Optional[Dict[str, Any]]:
    """
    Call Groq Structured Outputs with strict JSON-schema enforcement.
    Falls back to None on repeated API/schema failures.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=max(GROQ_TEMPERATURE, 1e-8),
                max_completion_tokens=GROQ_MAX_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Structured output attempt %d failed: %s", attempt, exc)
    return None


# ─── Step 5: Evidence validation ─────────────────────────────────────────────

def validate_biomarker_evidence(
    gene: str,
    disease_keyword: str,
    abstracts: List[Dict[str, str]],
    groq_client: Optional[Groq] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate literature evidence for a biomarker using Groq LLM.

    Parameters
    ----------
    gene             : gene identifier (e.g. ENSG or gene symbol)
    disease_keyword  : disease context string
    abstracts        : list of PubMed abstract dicts
    groq_client      : optional shared Groq client (created if None)

    Returns
    -------
    Parsed evidence dict or None if LLM call failed.
    """
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = groq_client or Groq(api_key=GROQ_API_KEY)
    abstract_text = _format_abstracts(abstracts)

    user_msg = (
        f"Gene: {gene}\nDisease context: {disease_keyword}\n\n"
        f"PubMed Abstracts:\n{abstract_text}\n\n"
        "Evaluate the evidence for this gene's role in the disease using ONLY the above abstracts."
    )

    result = _call_groq(client, EVIDENCE_SYSTEM_PROMPT, user_msg)
    if result is not None:
        result["gene"] = gene  # enforce correct gene ID
    return result


# ─── Step 6: Adversarial falsification ──────────────────────────────────────

def adversarial_falsify(
    gene: str,
    disease_keyword: str,
    abstracts: List[Dict[str, str]],
    groq_client: Optional[Groq] = None,
) -> Optional[Dict[str, Any]]:
    """
    Identify weaknesses and contradictions for a biomarker using Groq LLM (adversarial mode).

    Returns
    -------
    Parsed vulnerability dict or None if LLM call failed.
    """
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = groq_client or Groq(api_key=GROQ_API_KEY)
    abstract_text = _format_abstracts(abstracts)

    user_msg = (
        f"Gene: {gene}\nDisease context: {disease_keyword}\n\n"
        f"PubMed Abstracts:\n{abstract_text}\n\n"
        "Perform adversarial analysis — identify every methodological weakness, "
        "contradictory finding, small-sample issue, or correlation-only claim "
        "in the evidence above. Ground every criticism in specific text."
    )

    result = _call_groq(client, ADVERSARIAL_SYSTEM_PROMPT, user_msg)
    if result is not None:
        result["gene"] = gene
        # Ensure vulnerability_score is a proper float in [0,1]
        vs = result.get("vulnerability_score", 0.5)
        result["vulnerability_score"] = max(0.0, min(1.0, float(vs)))
    return result


# ─── Batch processing ────────────────────────────────────────────────────────

def run_full_llm_validation(
    gene_abstracts: Dict[str, List[Dict[str, str]]],
    disease_keyword: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Run both evidence validation and adversarial falsification for all genes.

    Parameters
    ----------
    gene_abstracts   : {gene: [abstract_dicts]}  — output of fetch_all_biomarkers
    disease_keyword  : disease context string (e.g. "breast cancer")

    Returns
    -------
    {gene: {"evidence": {...}, "adversarial": {...}}}
    """
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=GROQ_API_KEY)
    results: Dict[str, Dict[str, Any]] = {}

    for gene, abstracts in gene_abstracts.items():
        logger.info("LLM validation for gene: %s  (%d abstracts)", gene, len(abstracts))
        gene_result: Dict[str, Any] = {}

        # Step 5 — evidence scoring
        evidence = validate_biomarker_evidence(
            gene, disease_keyword, abstracts, groq_client=client
        )
        gene_result["evidence"] = evidence

        # Step 6 — adversarial falsification
        adversarial = adversarial_falsify(
            gene, disease_keyword, abstracts, groq_client=client
        )
        gene_result["adversarial"] = adversarial

        results[gene] = gene_result

    logger.info(
        "LLM validation complete: %d genes processed, %d with evidence, %d with adversarial",
        len(results),
        sum(1 for v in results.values() if v.get("evidence") is not None),
        sum(1 for v in results.values() if v.get("adversarial") is not None),
    )
    return results


BIOMED_AUDIT_SYSTEM_PROMPT = """SYSTEM ROLE:
You are an advanced biomedical AI system specialized in multi-omics analytics, disease risk modeling, biomarker validation, and therapeutic target discovery.

You must behave as a strict scientific auditor and not as a generic assistant.
You must avoid hallucinations.
If uncertain, explicitly say "Insufficient biological evidence available."

You must execute a full 10-stage audit exactly as specified:
1) Data quality audit
2) Feature selection validation
3) Multi-omics integration check
4) SHAP explainability audit
5) Biological literature validation
6) Causal inference reasoning
7) Therapeutic target mapping
8) Bias and generalization analysis
9) Biomarker confidence scoring
10) Final system verdict

IMPORTANT RULES:
- Never fabricate references.
- Never invent drug names.
- If uncertain, use: "Insufficient validated biological evidence."
- Be critical, not optimistic.
- Prioritize scientific integrity over positivity.
- Respond with JSON ONLY that matches the provided schema.
"""


def run_biomedical_system_audit(
    audit_input: Dict[str, Any],
    groq_client: Optional[Groq] = None,
) -> Dict[str, Any]:
    """
    Run the 10-stage biomedical audit workflow with strict JSON output.

    Expected input keys
    -------------------
    disease_name, multi_omics_features, selected_biomarkers, shap_importance_scores,
    stability_scores, model_metrics, training_sample_size, validation_sample_size
    """
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = groq_client or Groq(api_key=GROQ_API_KEY)

    biomarkers = list(audit_input.get("selected_biomarkers", []))
    shap_scores = dict(audit_input.get("shap_importance_scores", {}))
    stability_scores = dict(audit_input.get("stability_scores", {}))
    train_n = int(audit_input.get("training_sample_size", 0) or 0)
    val_n = int(audit_input.get("validation_sample_size", 0) or 0)

    multi = dict(audit_input.get("multi_omics_features", {}))
    all_features = (
        list(multi.get("genomics_features", []))
        + list(multi.get("transcriptomics_features", []))
        + list(multi.get("proteomics_features", []))
    )
    total_features = len(all_features)
    total_samples = max(train_n + val_n, 1)
    feature_sample_ratio = total_features / total_samples

    unstable = [g for g in biomarkers if float(stability_scores.get(g, 0.0)) < 0.5]

    user_payload = {
        "inputs": {
            "disease_name": audit_input.get("disease_name", ""),
            "training_sample_size": train_n,
            "validation_sample_size": val_n,
            "model_metrics": audit_input.get("model_metrics", {}),
            "selected_biomarkers": biomarkers[:10],
            "shap_importance_scores": {k: v for k, v in sorted(shap_scores.items(), key=lambda x: -float(x[1]))[:10]},
            "stability_scores": {k: v for k, v in stability_scores.items() if k in biomarkers[:10]},
        },
        "precomputed_checks": {
            "feature_count": total_features,
            "sample_count": total_samples,
            "feature_sample_ratio": round(feature_sample_ratio, 4),
            "rule_based_overfitting_risk": "HIGH" if feature_sample_ratio > 10 else "MODERATE or LOW",
            "flagged_unstable_features": unstable,
            "stability_thresholds": {
                "highly_stable": ">0.75",
                "moderately_stable": "0.50-0.75",
                "unstable": "<0.50",
            },
            "if_evidence_uncertain_use": "Insufficient validated biological evidence.",
        },
        "output_constraints": {
            "json_only": True,
            "do_not_invent_references": True,
            "do_not_invent_drugs": True,
            "biomarker_count_should_match_input": len(biomarkers),
        },
    }

    user_prompt = (
        "Run the full biomedical audit and return ONLY the JSON object matching the schema.\n"
        "Use the provided input payload and precomputed checks exactly as constraints:\n\n"
        f"{json.dumps(user_payload, ensure_ascii=False)}"
    )

    # Use JSON-object mode (most reliable with llama-3.3-70b-versatile)
    result = _call_groq(client, BIOMED_AUDIT_SYSTEM_PROMPT, user_prompt)
    if result is not None:
        return result

    raise RuntimeError("Groq biomedical audit failed after all attempts.")


# ─── Clinical Guidance Generation (Separate from Full Audit) ────────────────

CLINICAL_GUIDANCE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "when_to_visit_doctor": {"type": "string"},
        "symptoms_to_watch": {"type": "array", "items": {"type": "string"}},
        "lifestyle_recommendations": {"type": "array", "items": {"type": "string"}},
        "follow_up_timeline": {"type": "string"},
        "emergency_signs": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "when_to_visit_doctor",
        "symptoms_to_watch",
        "lifestyle_recommendations",
        "follow_up_timeline",
        "emergency_signs",
    ],
    "additionalProperties": False,
}

CLINICAL_GUIDANCE_SYSTEM_PROMPT = """You are a clinical guidance assistant providing actionable medical advice for patients.

You must provide practical, evidence-based clinical recommendations in JSON format only.

Base your guidance on:
- The disease/condition being assessed
- The predicted risk level (positive/negative and probability)
- The specific biomarkers identified (if any)
- Standard clinical guidelines for the disease

IMPORTANT RULES:
- Be conservative: recommend earlier medical consultation for higher risk
- Provide specific, actionable advice
- Never fabricate medical information
- Use clear, patient-friendly language
- Respond with JSON ONLY matching the schema

EXAMPLE OUTPUT FORMAT:
{
  "when_to_visit_doctor": "Schedule a consultation with your oncologist within 2-4 weeks to discuss these biomarker findings.",
  "symptoms_to_watch": [
    "Persistent cough lasting more than 3 weeks",
    "Unexplained weight loss",
    "Chest pain or discomfort"
  ],
  "lifestyle_recommendations": [
    "Avoid tobacco smoke and second hand smoke exposure",
    "Adopt a diet rich in fruits and vegetables",
    "Engage in moderate exercise 150 minutes per week"
  ],
  "follow_up_timeline": "Schedule follow-up imaging and blood work every 3-6 months",
  "emergency_signs": [
    "Severe chest pain or pressure",
    "Sudden severe shortness of breath",
    "Coughing up large amounts of blood"
  ]
}
"""


def generate_clinical_guidance(
    disease_name: str,
    prediction_label: str,
    prediction_prob: float,
    top_biomarkers: List[str] = None,
    groq_client: Optional[Groq] = None,
) -> Dict[str, Any]:
    """
    Generate clinical guidance for a patient based on prediction results.
    
    This is a simpler, separate function from the full biomedical audit.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set, returning default clinical guidance")
        return {
            "when_to_visit_doctor": "Consult with your healthcare provider to discuss these results",
            "symptoms_to_watch": ["Monitor for any unusual symptoms", "Track changes in your condition"],
            "lifestyle_recommendations": ["Maintain a healthy diet", "Exercise regularly", "Avoid smoking and excessive alcohol"],
            "follow_up_timeline": "Follow your healthcare provider's recommendations",
            "emergency_signs": ["Severe pain", "Difficulty breathing", "Sudden changes in consciousness"],
        }
    
    client = groq_client or Groq(api_key=GROQ_API_KEY)
    
    biomarker_context = ""
    if top_biomarkers:
        biomarker_context = f"\n\nTop biomarkers identified: {', '.join(top_biomarkers[:5])}"
    
    user_prompt = f"""Generate clinical guidance for a patient with the following assessment:

Disease: {disease_name}
Prediction: {prediction_label}
Risk Probability: {prediction_prob:.1%}{biomarker_context}

Provide practical clinical guidance in JSON format matching the schema.
Be specific to {disease_name} and the {prediction_label} prediction with {prediction_prob:.1%} probability.
"""
    
    # Use JSON-object mode (most reliable with llama-3.3-70b-versatile)
    result = _call_groq(client, CLINICAL_GUIDANCE_SYSTEM_PROMPT, user_prompt)
    if result is not None:
        return result
    
    # Ultimate fallback: return generic guidance
    logger.error("Clinical guidance generation failed, returning generic guidance")
    return {
        "when_to_visit_doctor": f"Consult with your healthcare provider about this {prediction_label} {disease_name} prediction",
        "symptoms_to_watch": ["Monitor for any unusual symptoms related to your condition"],
        "lifestyle_recommendations": ["Follow a healthy lifestyle", "Avoid known risk factors for your condition"],
        "follow_up_timeline": "Discuss appropriate follow-up schedule with your healthcare provider",
        "emergency_signs": ["Seek immediate medical care for severe or sudden symptoms"],
    }

