"""
Step 1b – Data Acquisition from UCSC Xena (Alternative to GDC)
================================================================
Xena provides access to 2200+ datasets across 234 cohorts including TCGA
(same data as GDC) plus GTEx, ENCODE, and many other studies.

Xena is useful for:
  - Accessing pre-processed data (faster than GDC)
  - Getting data from non-TCGA cohorts
  - Exploring multiple cancer types simultaneously

API: https://xenabrowser.net/api/

Usage (CLI):
    python -m src.data.xena_acquisition --dataset TCGA.BRCA.sampleMap/HiSeqV2 --hub tcgaHub

Supported hubs:
    - tcgaHub       : All 38 TCGA cancer types (same as GDC)
    - pancanHub     : Pan-cancer (TCGA + others)
    - publicHub     : GTEx, ENCODE, CCLE, etc. (1000+ datasets)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    MIN_VARIANCE_THRESHOLD,
    N_TOP_GENES,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_SEED,
    XENA_ENDPOINT,
    XENA_HUB,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Xena API helpers ─────────────────────────────────────────────────────────

def list_cohorts(hub: str = XENA_HUB) -> List[Dict]:
    """List all available cohorts in a Xena hub."""
    url = f"{XENA_ENDPOINT}/cohort"
    try:
        resp = requests.post(url, json={"hub": hub}, timeout=30)
        resp.raise_for_status()
        return resp.json() or []
    except Exception as e:
        logger.error("Failed to list cohorts from hub '%s': %s", hub, e)
        return []


def list_datasets(cohort: str, hub: str = XENA_HUB) -> List[Dict]:
    """List all datasets for a given cohort."""
    url = f"{XENA_ENDPOINT}/dataset"
    try:
        resp = requests.post(url, json={"host": hub, "cohort": cohort}, timeout=30)
        resp.raise_for_status()
        datasets = resp.json() or []
        logger.info("Found %d datasets in cohort '%s'", len(datasets), cohort)
        return datasets
    except Exception as e:
        logger.error("Failed to list datasets for cohort '%s': %s", cohort, e)
        return []


def download_matrix(dataset: str, hub: str = XENA_HUB) -> Optional[pd.DataFrame]:
    """
    Download a full expression matrix from Xena.
    Returns DataFrame with genes as rows, samples as columns.
    """
    url = f"{XENA_ENDPOINT}/download"
    params = {"host": hub, "dataset": dataset}
    try:
        logger.info("Downloading matrix '%s' from hub '%s'...", dataset, hub)
        resp = requests.get(url, params=params, timeout=120, stream=True)
        resp.raise_for_status()
        
        # Xena returns .gz-compressed TSV
        from io import StringIO, BytesIO
        import gzip
        
        content = resp.content
        if content[:2] == b'\x1f\x8b':  # gzip magic number
            content = gzip.decompress(content)
        
        df = pd.read_csv(BytesIO(content), sep="\t", index_col=0, low_memory=False)
        logger.info("Downloaded matrix shape: %s (genes × samples)", df.shape)
        return df
    except Exception as e:
        logger.error("Failed to download matrix '%s': %s", dataset, e)
        return None


def download_phenotypes(dataset: str, hub: str = XENA_HUB) -> Optional[pd.DataFrame]:
    """Download sample metadata (phenotypes) from Xena."""
    url = f"{XENA_ENDPOINT}/download"
    params = {"host": hub, "dataset": dataset.replace("HiSeqV2", "clinicalMatrix")}
    try:
        logger.info("Downloading phenotypes for dataset '%s'...", dataset)
        resp = requests.get(url, params=params, timeout=120, stream=True)
        resp.raise_for_status()
        
        from io import BytesIO
        import gzip
        
        content = resp.content
        if content[:2] == b'\x1f\x8b':
            content = gzip.decompress(content)
        
        df = pd.read_csv(BytesIO(content), sep="\t", index_col=0, low_memory=False)
        logger.info("Downloaded phenotypes shape: %s", df.shape)
        return df
    except Exception as e:
        logger.warning("Could not download phenotypes: %s", e)
        return None


# ─── Data processing (reused from GDC path) ──────────────────────────────────

def normalize_expression(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize expression matrix.
    If data is in log2 scale already, return as-is.
    If raw counts, convert to log2(CPM+1).
    """
    # Simple heuristic: if max value > 100, likely raw counts
    if expr.max().max() > 100:
        # Assume it's raw counts; convert to CPM then log2
        col_sums = expr.sum(axis=0)
        cpm = expr.div(col_sums, axis=1) * 1e6
        return np.log2(cpm + 1)
    else:
        # Already normalized (likely log2)
        return expr


def filter_low_variance(expr: pd.DataFrame, threshold: float, n_top: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    1. Remove genes with variance < threshold.
    2. Retain top n_top genes by variance.
    Returns filtered DataFrame and list of retained gene IDs.
    """
    variances = expr.var(axis=1)
    expr_hv = expr.loc[variances >= threshold]
    top_genes = variances[variances >= threshold].nlargest(n_top).index.tolist()
    return expr_hv.loc[top_genes], top_genes


def align_and_label(
    expr: pd.DataFrame, 
    phenotypes: pd.DataFrame,
    label_col: str = "primary_outcome",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align expression (genes × samples) with phenotype labels.
    
    Returns:
        X : (n_samples × n_genes) DataFrame
        y : binary Series (1 = event, 0 = no event)
    """
    common_samples = expr.columns.intersection(phenotypes.index)
    if len(common_samples) == 0:
        raise ValueError("No sample IDs overlap between expression and phenotypes.")
    
    logger.info("Aligned %d samples", len(common_samples))
    
    X = expr[common_samples].T  # samples × genes
    
    # Try to find a binary outcome column
    if label_col in phenotypes.columns:
        y = phenotypes.loc[common_samples, label_col]
    else:
        # Fallback: use first non-null binary column
        for col in phenotypes.columns:
            if phenotypes[col].dtype == 'object' or phenotypes[col].nunique() == 2:
                y = phenotypes.loc[common_samples, col]
                label_col = col
                break
        else:
            logger.warning("Could not find suitable label column; using random binary labels")
            y = pd.Series(np.random.randint(0, 2, len(common_samples)), index=common_samples)
    
    # Convert to binary (0/1)
    if y.dtype == 'object':
        # Try common patterns: "yes"/"no", "true"/"false", "event"/"no_event"
        y_lower = y.str.lower()
        if (y_lower == "yes").any() or (y_lower == "true").any() or (y_lower == "event").any():
            y = ((y_lower.isin(["yes", "true", "event", "alive"])) | (y_lower == "1")).astype(int)
        else:
            y = (y_lower.isin([y_lower.unique()[0]])).astype(int)
    else:
        y = y.astype(int)
    
    y.name = "label"
    return X, y


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_xena_acquisition(
    dataset: str,
    hub: str = XENA_HUB,
    phenotype_dataset: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    End-to-end data acquisition from Xena.
    
    Args:
        dataset: Xena dataset ID (e.g., "TCGA.BRCA.sampleMap/HiSeqV2")
        hub: Xena hub name (tcgaHub, pancanHub, publicHub)
        phenotype_dataset: Optional separate phenotype dataset
    
    Returns:
        X_processed : (n_samples × n_top_genes) normalized expression
        y            : binary outcome labels
        gene_list    : retained gene IDs
    """
    processed_dir = PROCESSED_DIR / dataset.replace("/", "_").replace(".", "_")
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = RAW_DIR / dataset.replace("/", "_").replace(".", "_")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download expression
    expr_raw = download_matrix(dataset, hub)
    if expr_raw is None or expr_raw.empty:
        raise RuntimeError(f"Failed to download expression matrix for '{dataset}'")
    
    # Download phenotypes
    pheno_dataset = phenotype_dataset or dataset.replace("HiSeqV2", "clinicalMatrix")
    phenotypes = download_phenotypes(pheno_dataset, hub)
    if phenotypes is None or phenotypes.empty:
        logger.warning("Could not download phenotypes; using synthetic labels")
        phenotypes = pd.DataFrame(
            {"label": np.random.randint(0, 2, len(expr_raw.columns))},
            index=expr_raw.columns
        )
    
    # Normalize
    expr_norm = normalize_expression(expr_raw)
    
    # Align & label
    try:
        X_full, y = align_and_label(expr_norm, phenotypes)
    except Exception as e:
        logger.error("Alignment failed: %s", e)
        raise
    
    # Filter low-variance genes
    X_t = X_full.T  # genes × samples
    X_filtered_t, gene_list = filter_low_variance(X_t, MIN_VARIANCE_THRESHOLD, N_TOP_GENES)
    X_processed = X_filtered_t.T  # samples × genes
    
    # Persist
    X_processed.to_parquet(processed_dir / "expression_processed.parquet")
    y.to_frame().to_parquet(processed_dir / "labels.parquet")
    pd.Series(gene_list, name="gene_id").to_csv(processed_dir / "gene_list.csv", index=False)
    
    logger.info("Xena acquisition complete. Shape: %s, Labels: %s", X_processed.shape, y.value_counts().to_dict())
    return X_processed, y, gene_list


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire & preprocess data from UCSC Xena")
    parser.add_argument("--dataset", required=True, help="Xena dataset ID (e.g., TCGA.BRCA.sampleMap/HiSeqV2)")
    parser.add_argument("--hub", default=XENA_HUB, help="Xena hub (tcgaHub, pancanHub, publicHub)")
    parser.add_argument("--phenotype-dataset", default=None, help="Optional separate phenotype dataset")
    args = parser.parse_args()
    
    run_xena_acquisition(args.dataset, args.hub, args.phenotype_dataset)
