#!/usr/bin/env python3
"""Generate patient-facing reports from model predictions only.

Produces non-prescriptive, general guidance and resource links based only on
predicted probabilities. Includes a clear medical disclaimer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def risk_category(p: float) -> str:
    if p < 0.2:
        return "Low"
    if p < 0.5:
        return "Moderate"
    return "High"


def generate_report_row(sample_id: str, prob: float) -> dict:
    pct = round(float(prob) * 100, 1)
    cat = risk_category(prob)
    report = {
        "sample_id": str(sample_id),
        "predicted_probability": float(prob),
        "predicted_percent": pct,
        "risk_category": cat,
        "interpretation": (
            f"Estimated chance: {pct}% — categorized as {cat} risk."
        ),
        "recommended_next_steps": [],
        "suggested_specialists": ["Primary care physician", "Medical oncologist"],
        "resources": {
            "cancer_information": "https://www.cancer.gov/",
            "patient_support": "https://www.cancer.org/",
            "clinical_trials": "https://clinicaltrials.gov/",
        },
        "disclaimer": (
            "This report is informational only and not medical advice. "
            "Discuss results with a licensed healthcare professional for diagnosis and treatment."
        ),
    }

    # Recommendations based on risk category (general, non-prescriptive)
    if cat == "Low":
        report["recommended_next_steps"].append("Discuss results with your primary care provider if concerned.")
        report["recommended_next_steps"].append("Maintain routine screening and healthy lifestyle (smoking cessation, balanced diet, exercise).")
    elif cat == "Moderate":
        report["recommended_next_steps"].append("Schedule follow-up with your primary care provider to review clinical context and consider diagnostic testing.")
        report["recommended_next_steps"].append("If clinically indicated, referral to a specialist (e.g., oncologist) for diagnostic workup may be appropriate.")
    else:  # High
        report["recommended_next_steps"].append("Contact your healthcare provider promptly to discuss confirmatory diagnostic tests (imaging, biopsy) and referral to oncology.")
        report["recommended_next_steps"].append("Consider evaluation at a comprehensive cancer center for multidisciplinary care and treatment planning.")
        report["suggested_specialists"].extend(["Surgical oncologist", "Radiation oncologist", "Medical oncologist"])

    # Prevention / general supportive measures (high-level)
    report["prevention_and_support"] = [
        "Avoid tobacco and excessive alcohol consumption.",
        "Follow age-appropriate cancer screening guidelines.",
        "Discuss vaccination options with your provider (e.g., HPV vaccine where applicable).",
    ]

    # Safety note about 'cure'
    report["note_on_cure"] = (
        "Treatment and outcomes depend on diagnosis, disease stage, and individual health. "
        "Only a licensed clinician can recommend appropriate therapy; this report does not claim cures."
    )

    return report


def main(project: str):
    repo = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(repo))
    from config.settings import RESULTS_DIR

    preds_path = Path(RESULTS_DIR) / project / "predictions_test.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    preds = pd.read_csv(preds_path)
    out_dir = Path(RESULTS_DIR) / project / "patient_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for _, row in preds.iterrows():
        rpt = generate_report_row(row.get("sample_id", "unknown"), float(row["y_prob"]))
        reports.append(rpt)
        # write per-sample report (JSON) without raw features
        sid = str(row.get("sample_id", "sample"))
        safe_name = sid.replace('/', '_')
        (out_dir / f"report_{safe_name}.json").write_text(json.dumps(rpt, indent=2), encoding="utf-8")

    # write aggregated summary
    agg = {
        "project": project,
        "n_reports": len(reports),
        "reports": reports,
    }
    (out_dir / "reports_index.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    args = parser.parse_args()
    main(args.project)
