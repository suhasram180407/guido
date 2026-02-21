#!/usr/bin/env python3
"""Predict one test sample (real data) and send prediction-only payload to Groq.

Does NOT send raw feature values or sample identifiers. Sends a compact
predicted-probability summary to the Groq biomedical audit pipeline and prints
the resulting report.
"""
from __future__ import annotations
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(project: str):
    repo = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(repo))

    import joblib
    from src.api.pipeline import _load_splits, run_biomedical_audit_pipeline
    from config.settings import MODELS_DIR, RESULTS_DIR

    model_path = MODELS_DIR / project / "primary_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model = joblib.load(model_path)
    splits = _load_splits(project)
    X_test, y_test = splits["test"]

    if X_test.shape[0] == 0:
        raise RuntimeError("Test split is empty")

    # pick the first test sample (real data) — but do NOT include features
    sample_idx = X_test.index[0]

    try:
        y_prob = float(model.predict_proba(X_test.loc[[sample_idx]])[:, 1][0])
    except Exception:
        # fallback
        y_pred = int(model.predict(X_test.loc[[sample_idx]])[0])
        y_prob = float(y_pred)

    # Build a prediction-only summary (single-sample)
    pred_summary = {
        "n_samples": 1,
        "mean_prob": y_prob,
        "std_prob": 0.0,
        "pos_count": int(y_prob >= 0.5),
        "neg_count": int(y_prob < 0.5),
        "prob_hist_bins": [1 if y_prob <= 0.2 else 0, 0, 0, 0, 0],
    }

    # Attach existing eval metrics if available
    metrics_path = Path(RESULTS_DIR) / project / "eval_test.json"
    model_metrics = {}
    if metrics_path.exists():
        try:
            model_metrics = json.loads(metrics_path.read_text())
        except Exception:
            model_metrics = {}

    audit_input = {
        "project_id": project,
        "disease_name": "lung adenocarcinoma",
        "predictions_summary": pred_summary,
        "model_metrics": model_metrics,
        # intentionally omit raw features and IDs
        "training_sample_size": int(X_test.shape[0]) if hasattr(X_test, 'shape') else 0,
        "validation_sample_size": 0,
        "selected_biomarkers": [],
        "shap_importance_scores": {},
        "stability_scores": {},
    }

    logger.info("Sending single-sample prediction summary to Groq for project %s", project)
    report = run_biomedical_audit_pipeline(audit_input)
    out_path = Path(RESULTS_DIR) / project / "biomedical_audit_prediction_only.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    args = parser.parse_args()
    main(args.project)
