"""
Synthetic data generator for local development and testing.
Produces a realistic multi-omics-style dataset WITHOUT requiring TCGA access.

Usage:
    python -m src.data.synthetic_data --project TCGA-BRCA --n-samples 400
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    N_TOP_GENES,
    PROCESSED_DIR,
    RANDOM_SEED,
    TCGA_PROJECT_ID,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_synthetic_dataset(
    n_samples: int = 400,
    n_genes: int = N_TOP_GENES,
    n_signal_genes: int = 50,
    pos_rate: float = 0.40,
    project_id: str = TCGA_PROJECT_ID,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Generate synthetic log2(TPM+1) expression matrix with embedded signal.

    n_signal_genes genes have expression differences between classes,
    representing biologically meaningful variation. The rest is noise.
    """
    rng = np.random.default_rng(seed)

    # Gene IDs using ENSG-style naming
    gene_ids = [f"ENSG{str(i).zfill(11)}" for i in range(n_genes)]

    # Sample IDs using TCGA-style naming
    sample_ids = [f"TCGA-XX-{str(i).zfill(4)}-01A" for i in range(n_samples)]

    # Binary labels
    y_arr = rng.choice([0, 1], size=n_samples, p=[1 - pos_rate, pos_rate])

    # Base expression (log2TPM-like: range ~0-15)
    X = rng.normal(loc=6.0, scale=2.0, size=(n_samples, n_genes))

    # Inject signal into first n_signal_genes
    signal_idx = np.arange(n_signal_genes)
    for i in range(n_signal_genes):
        effect = rng.uniform(1.0, 3.0)  # fold-change effect
        X[y_arr == 1, i] += effect

    # Clip to valid log2TPM range [0, 18]
    X = np.clip(X, 0, 18)

    X_df = pd.DataFrame(X, index=sample_ids, columns=gene_ids, dtype=np.float32)
    y_series = pd.Series(y_arr.astype(int), index=sample_ids, name="label")

    # Persist
    out_dir = PROCESSED_DIR / project_id
    out_dir.mkdir(parents=True, exist_ok=True)
    X_df.to_parquet(out_dir / "rnaseq_processed.parquet")
    y_series.to_frame().to_parquet(out_dir / "labels.parquet")
    pd.Series(gene_ids, name="gene_id").to_csv(out_dir / "gene_list.csv", index=False)
    logger.info(
        "Synthetic dataset saved to %s  shape=%s  pos_rate=%.3f",
        out_dir, X_df.shape, y_series.mean(),
    )
    return X_df, y_series, gene_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic multi-omics dataset")
    parser.add_argument("--project", default=TCGA_PROJECT_ID)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-genes", type=int, default=N_TOP_GENES)
    args = parser.parse_args()
    generate_synthetic_dataset(n_samples=args.n_samples, n_genes=args.n_genes, project_id=args.project)
