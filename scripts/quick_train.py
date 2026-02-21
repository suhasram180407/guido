#!/usr/bin/env python3
"""Quick train using partial data for fast demos.

Usage:
  python scripts/quick_train.py --project TCGA-LUAD --n-samples 50 --n-genes 500
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import RAW_DIR, MODELS_DIR, RESULTS_DIR
from src.data.acquisition import fpkm_to_log2tpm
from src.data.splitter import split_data
from src.models.baselines import run_baselines
from src.models.primary_model import build_elasticnet, train_primary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def quick_train(project: str, n_samples: int = 50, n_genes: int = 500):
    raw_dir = Path(RAW_DIR) / project
    raw_path = raw_dir / f"{project}_rnaseq_raw.parquet"
    clin_path = raw_dir / f"{project}_clinical.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw matrix not found: {raw_path}")

    logger.info("Loading raw matrix %s", raw_path)
    expr = pd.read_parquet(raw_path)
    logger.info("Raw shape: %s", expr.shape)

    # Coerce numeric and normalize
    expr_norm = fpkm_to_log2tpm(expr)

    # sample columns (samples are columns)
    samples = list(expr_norm.columns)
    if len(samples) > n_samples:
        sampled = np.random.RandomState(42).choice(samples, size=n_samples, replace=False)
        expr_sub = expr_norm.loc[:, sampled]
    else:
        expr_sub = expr_norm

    # get labels
    if clin_path.exists():
        clin = pd.read_parquet(clin_path)
        # align by submitter id if available; else create random
        common = expr_sub.columns.intersection(clin.index)
        if len(common) >= 2:
            y = clin.loc[common, 'vital_status'].str.lower() == 'dead'
            y = y.astype(int)
            X = expr_sub.loc[:, common].T
        else:
            logger.warning("Clinical missing or no overlap — generating balanced random labels")
            X = expr_sub.T
            idx = X.index
            y = pd.Series(np.random.randint(0, 2, size=len(idx)), index=idx)
    else:
        logger.warning("Clinical file missing — generating balanced random labels")
        X = expr_sub.T
        idx = X.index
        # make balanced
        half = len(idx) // 2
        labels = np.array([1]*half + [0]*(len(idx)-half))
        np.random.RandomState(42).shuffle(labels)
        y = pd.Series(labels, index=idx)

    # reduce genes by variance
    variances = X.var(axis=0)
    top_genes = variances.nlargest(n_genes).index.tolist()
    X = X.loc[:, top_genes]

    logger.info("Training dataset: samples=%d genes=%d", X.shape[0], X.shape[1])

    # split
    splits = split_data(X, y)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    # baselines
    logger.info("Running quick baselines (small CV)...")
    baseline_results = run_baselines(X_train, y_train)

    # primary model (ElasticNet)
    logger.info("Training primary ElasticNet model (quick)...")
    model, val_metrics, gate_passed = train_primary(
        X_train, y_train, X_val, y_val, baseline_results, use_neural_net=False, save_path=Path(MODELS_DIR)/project
    )

    # ensure primary_model.joblib exists at expected path
    import joblib
    joblib.dump(model, Path(MODELS_DIR)/project/"primary_model.joblib")

    # save splits (so evaluator can load)
    splits_dir = Path(MODELS_DIR)/project/'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(splits_dir / 'X_train.parquet')
    y_train.to_frame().to_parquet(splits_dir / 'y_train.parquet')
    X_val.to_parquet(splits_dir / 'X_val.parquet')
    y_val.to_frame().to_parquet(splits_dir / 'y_val.parquet')
    X_test.to_parquet(splits_dir / 'X_test.parquet')
    y_test.to_frame().to_parquet(splits_dir / 'y_test.parquet')

    # run evaluation quickly
    from src.api.pipeline import run_evaluation_pipeline
    metrics = run_evaluation_pipeline(project, 'test')
    out_dir = Path(RESULTS_DIR) / project
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'eval_test.json').write_text(json.dumps(metrics, indent=2))

    logger.info("Quick training complete. Eval saved to %s", out_dir / 'eval_test.json')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--n-samples', type=int, default=50)
    parser.add_argument('--n-genes', type=int, default=500)
    args = parser.parse_args()
    quick_train(args.project, n_samples=args.n_samples, n_genes=args.n_genes)
