#!/usr/bin/env python3
"""Send aggregated model predictions (no raw data) to Groq for audit generation.

This script reads `results/<project>/predictions_test.csv`, computes a compact
predictions summary (no sample-level identifiers), and calls the Groq audit
workflow to generate a biomedical audit report.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_summary(preds: pd.DataFrame) -> dict:
    probs = preds["y_prob"].astype(float).values
    preds_bin = preds["y_pred"].astype(int).values
    n = len(probs)
    mean_prob = float(np.mean(probs))
    std_prob = float(np.std(probs))
    pos_count = int(preds_bin.sum())
    neg_count = int(n - pos_count)
    # histogram with 5 bins
    hist, edges = np.histogram(probs, bins=5, range=(0.0, 1.0))
    hist_bins = [int(x) for x in hist]
    return {
        "n_samples": int(n),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "prob_hist_bins": hist_bins,
        "prob_hist_edges": [float(e) for e in edges],
    }


def main(project: str, disease: str = "lung adenocarcinoma"):
    repo = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(repo))

    from config.settings import RESULTS_DIR
    from src.api.pipeline import run_biomedical_audit_pipeline

    preds_path = Path(RESULTS_DIR) / project / "predictions_test.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    preds = pd.read_csv(preds_path)
    # remove any sample identifiers before sending
    summary = make_summary(preds)

    # load existing eval metrics if present
    eval_path = Path(RESULTS_DIR) / project / "eval_test.json"
    model_metrics = {}
    if eval_path.exists():
        try:
            model_metrics = json.loads(eval_path.read_text())
        except Exception:
            model_metrics = {}

    audit_input = {
        "project_id": project,
        "disease_name": disease,
        "predictions_summary": summary,  # no sample-level data included
        "model_metrics": model_metrics,
        # intentionally omit `multi_omics_features` and raw feature values
        "training_sample_size": 0,
        "validation_sample_size": 0,
        "selected_biomarkers": [],
        "shap_importance_scores": {},
        "stability_scores": {},
    }

    logger.info("Sending predictions summary to Groq for project %s", project)
    report = run_biomedical_audit_pipeline(audit_input)
    out_path = Path(RESULTS_DIR) / project / f"biomedical_audit_from_predictions_{disease.replace(' ','_')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved Groq audit (predictions-only) to %s", out_path)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--disease', default='lung adenocarcinoma')
    args = parser.parse_args()
    main(args.project, args.disease)
