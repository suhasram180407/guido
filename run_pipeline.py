"""
CLI pipeline runner — execute the full end-to-end pipeline from the terminal.

Usage examples:
    # Full pipeline with synthetic data
    python run_pipeline.py --mode full --synthetic

    # Training only
    python run_pipeline.py --mode train --synthetic --project TCGA-BRCA

    # Biomarker analysis only (requires prior training run)
    python run_pipeline.py --mode biomarkers --project TCGA-BRCA

    # Start the API server
    python run_pipeline.py --mode api
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.utils import configure_logging, fix_seed
from config.settings import RANDOM_SEED, TCGA_PROJECT_ID

configure_logging()
fix_seed(RANDOM_SEED)
logger = logging.getLogger(__name__)


def mode_train(args):
    from src.api.pipeline import run_training_pipeline
    result = run_training_pipeline(
        project_id=args.project,
        use_synthetic=args.synthetic,
        use_neural_net=args.neural_net,
        n_synthetic_samples=args.n_samples,
        data_source=args.data_source,
        xena_dataset=args.xena_dataset,
    )
    logger.info("Training complete. Gate passed: %s", result["gate_passed"])
    logger.info("Stable biomarkers (%d): %s", len(result["stable_biomarkers"]), result["stable_biomarkers"][:5])
    return result


def mode_biomarkers(args):
    from src.api.pipeline import run_biomarker_pipeline
    result = run_biomarker_pipeline(project_id=args.project)
    logger.info(
        "Biomarker analysis complete. Total: %d, High-confidence: %d",
        result["total_biomarkers"],
        result["high_confidence_count"],
    )
    return result


def mode_evaluate(args):
    from src.api.pipeline import run_evaluation_pipeline
    result = run_evaluation_pipeline(project_id=args.project, split=args.split)
    logger.info("Evaluation on %s: %s", args.split, result)
    return result


def mode_audit(args):
    import json
    from src.api.pipeline import run_biomedical_audit_pipeline

    if not args.audit_input:
        raise ValueError("--audit-input is required for mode=audit")

    payload = json.loads(Path(args.audit_input).read_text(encoding="utf-8"))
    payload.setdefault("project_id", args.project)
    result = run_biomedical_audit_pipeline(payload)
    logger.info("Biomedical audit complete. Biomarkers audited: %d", len(result.get("biomarker_analysis", [])))
    return result


def mode_api(args):
    import uvicorn
    from config.settings import API_HOST, API_PORT
    logger.info("Starting FastAPI server on %s:%d", API_HOST, API_PORT)
    uvicorn.run("src.api.app:app", host=API_HOST, port=API_PORT, reload=False)


def mode_full(args):
    logger.info("=== FULL PIPELINE START ===")
    train_result = mode_train(args)

    if not train_result["gate_passed"]:
        logger.warning("Gate check failed — continuing anyway with fallback model.")

    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set — skipping biomarker LLM analysis step.")
    else:
        mode_biomarkers(args)

    mode_evaluate(args)
    logger.info("=== FULL PIPELINE COMPLETE ===")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-omics disease risk modeling pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "biomarkers", "evaluate", "audit", "api"],
        default="full",
    )
    parser.add_argument("--project", default=TCGA_PROJECT_ID, help="Project identifier")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic data (no real download)")
    parser.add_argument("--real-data", dest="synthetic", action="store_false",
                        help="Download real data (from GDC or Xena)")
    parser.add_argument("--data-source", choices=["gdc", "xena"], default="gdc",
                        help="Data source: 'gdc' (TCGA) or 'xena' (2200+ datasets)")
    parser.add_argument("--xena-dataset", default=None,
                        help="Xena dataset ID (required if --data-source=xena). E.g., TCGA.BRCA.sampleMap/HiSeqV2")
    parser.add_argument("--neural-net", action="store_true", default=False,
                        help="Use shallow neural net as primary model")
    parser.add_argument("--n-samples", type=int, default=400,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Evaluation split (for mode=evaluate)")
    parser.add_argument("--audit-input", default=None,
                        help="Path to JSON payload for mode=audit")

    args = parser.parse_args()

    dispatch = {
        "full":       mode_full,
        "train":      mode_train,
        "biomarkers": mode_biomarkers,
        "evaluate":   mode_evaluate,
        "audit":      mode_audit,
        "api":        mode_api,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
