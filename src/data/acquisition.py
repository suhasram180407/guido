"""
Step 1 – Data Acquisition from GDC (TCGA)
=========================================
Downloads RNA-seq HTSeq-FPKM-UQ counts and clinical metadata from the
GDC Data Portal REST API, aligns patient IDs, converts to log2(TPM+1),
filters low-variance genes, and serialises processed matrices to disk.

Usage (CLI):
    python -m src.data.acquisition --project TCGA-BRCA --out data/raw

Outputs:
    data/raw/<project>_rnaseq_raw.parquet
    data/raw/<project>_clinical.parquet
    data/processed/<project>_rnaseq_processed.parquet
    data/processed/<project>_labels.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    GDC_ENDPOINT,
    MIN_VARIANCE_THRESHOLD,
    N_TOP_GENES,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_SEED,
    TCGA_PROJECT_ID,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── GDC helpers ─────────────────────────────────────────────────────────────

def _gdc_files_query(project_id: str, workflow: str = "HTSeq - FPKM-UQ") -> List[Dict]:
    """
    Return list of GDC file metadata dicts for RNA-seq files.
    
    Flexible approach: finds Gene Expression Quantification files without strict
    workflow type matching, since modern GDC uses different file structures.
    """
    url = f"{GDC_ENDPOINT}/files"
    
    # Simple query: just get gene expression files for this project
    # Modern GDC files may not have workflow_type, so we fetch all and filter by filename
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "data_format", "value": "TSV"}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id",
        "format": "JSON",
        "size": "2000",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        hits = resp.json()["data"]["hits"]
        
        # Filter for standard expression file types
        # Accept: HTSeq counts, STAR counts, RSEM, or other quantified expression
        filtered = [
            h for h in hits
            if any(x in h['file_name'].lower() for x in [
                'gencode', 'counts', 'fpkm', 'tpm', 'rsem', 'expected_count'
            ])
        ]
        
        if not filtered:
            filtered = hits  # Fallback: use all if no matches on filename
        
        logger.info("Found %d RNA-seq files for project %s", len(filtered), project_id)
        return filtered
    except Exception as e:
        logger.error(f"Failed to query files for {project_id}: {e}")
        return []


def _download_file(file_id: str, dest: Path) -> Optional[Path]:
    """Download a single GDC file; return path or None on failure."""
    url = f"{GDC_ENDPOINT}/data/{file_id}"
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except Exception as exc:
        logger.warning("Failed to download %s: %s", file_id, exc)
        return None


def download_rnaseq(project_id: str, raw_dir: Path, max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Download all RNA-seq files for a TCGA project.

    Returns a DataFrame with shape (n_genes, n_samples) indexed by Ensembl gene IDs.
    Also writes raw parquet to disk.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    hits = _gdc_files_query(project_id)
    if max_files:
        hits = hits[:max_files]

    expr_dict: Dict[str, pd.Series] = {}
    for hit in tqdm(hits, desc="Downloading RNA-seq"):
        fid = hit["file_id"]
        fname = hit["file_name"]
        case_id = hit["cases"][0]["submitter_id"] if hit.get("cases") else fid

        dest = raw_dir / fname
        if not dest.exists():
            _download_file(fid, dest)
            time.sleep(0.1)  # be polite to GDC

        if dest.exists():
            try:
                df_file = pd.read_csv(dest, sep="\t", index_col=0, header=None, comment="#")
                expr_dict[case_id] = df_file.iloc[:, 0]
            except Exception as exc:
                logger.warning("Could not parse %s: %s", dest, exc)

    if not expr_dict:
        raise RuntimeError("No RNA-seq files downloaded. Check project ID and network access.")

    expr_matrix = pd.DataFrame(expr_dict)  # genes × samples
    out_path = raw_dir / f"{project_id}_rnaseq_raw.parquet"
    expr_matrix.to_parquet(out_path)
    logger.info("Saved raw expression matrix %s to %s", expr_matrix.shape, out_path)
    return expr_matrix


def download_clinical(project_id: str, raw_dir: Path) -> pd.DataFrame:
    """Download clinical metadata for all cases in a TCGA project."""
    url = f"{GDC_ENDPOINT}/cases"
    filters = {
        "op": "=",
        "content": {"field": "project.project_id", "value": project_id},
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "submitter_id,demographic.vital_status,diagnoses.age_at_diagnosis,demographic.gender",
        "format": "JSON",
        "size": "2000",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]

    records = []
    for h in hits:
        vital = "Unknown"
        if h.get("demographic"):
            vital = h["demographic"].get("vital_status", "Unknown")
        age = None
        if h.get("diagnoses"):
            age = h["diagnoses"][0].get("age_at_diagnosis")
        gender = h.get("demographic", {}).get("gender", "Unknown")
        records.append(
            {"patient_id": h["submitter_id"], "vital_status": vital, "age_at_diagnosis": age, "gender": gender}
        )

    clin = pd.DataFrame(records).set_index("patient_id")
    out_path = raw_dir / f"{project_id}_clinical.parquet"
    clin.to_parquet(out_path)
    logger.info("Saved clinical metadata %s to %s", clin.shape, out_path)
    return clin


# ─── Normalisation & preprocessing ─────────────────────────────────────────

def fpkm_to_log2tpm(fpkm_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Convert FPKM-UQ → TPM → log2(TPM+1).

    TPM_i = (FPKM_i / Σ FPKM) × 1e6
    """
    # Ensure numeric values; coerce non-numeric to NaN then treat as zero
    fpkm_numeric = fpkm_matrix.apply(pd.to_numeric, errors="coerce").fillna(0)
    col_sums = fpkm_numeric.sum(axis=0)
    # avoid divide-by-zero
    col_sums[col_sums == 0] = 1
    tpm = fpkm_numeric.div(col_sums, axis=1) * 1e6
    return np.log2(tpm + 1)


def filter_low_variance(expr: pd.DataFrame, threshold: float, n_top: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    1. Remove genes with variance < threshold.
    2. Retain top n_top genes by variance.
    Returns filtered DataFrame and list of retained gene IDs.
    """
    variances = expr.var(axis=1)

    # Genes meeting the variance threshold
    high_var = variances[variances >= threshold]

    # If no genes pass the threshold, fallback to selecting top n_top by variance
    if high_var.empty:
        top_genes = variances.nlargest(max(1, n_top)).index.tolist()
        filtered = expr.loc[top_genes]
        return filtered, top_genes

    # Otherwise, take top n_top from those that passed
    top_genes = high_var.nlargest(max(1, n_top)).index.tolist()
    filtered = expr.loc[top_genes]
    return filtered, top_genes


def align_and_label(expr: pd.DataFrame, clinical: pd.DataFrame, label_col: str = "vital_status") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align expression (genes × patients) with clinical labels.
    Returns:
        X : (n_samples × n_genes) DataFrame
        y : binary Series (1 = Dead, 0 = Alive)
    """
    common_patients = expr.columns.intersection(clinical.index)
    if len(common_patients) == 0:
        raise ValueError("No patient IDs overlap between expression and clinical data.")
    logger.info("Aligned %d patients", len(common_patients))

    X = expr[common_patients].T  # samples × genes
    y_raw = clinical.loc[common_patients, label_col]
    y = (y_raw.str.lower() == "dead").astype(int)
    y.name = "label"
    return X, y


# ─── Full pipeline entry point ────────────────────────────────────────────

def run_acquisition(project_id: str = TCGA_PROJECT_ID, max_files: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    End-to-end data acquisition and preprocessing.

    Returns:
        X_processed : (n_samples × n_top_genes) log2TPM expression
        y            : binary survival labels
        gene_list    : list of retained gene IDs
    """
    raw_dir = RAW_DIR / project_id
    processed_dir = PROCESSED_DIR / project_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Download
    expr_raw = download_rnaseq(project_id, raw_dir, max_files=max_files)
    clinical = download_clinical(project_id, raw_dir)

    # Normalise
    expr_log2tpm = fpkm_to_log2tpm(expr_raw)

    # Align & label
    X_full, y = align_and_label(expr_log2tpm, clinical)

    # Filter variance  (transpose back to genes × samples for variance calc)
    X_t = X_full.T  # genes × samples
    X_filtered_t, gene_list = filter_low_variance(X_t, MIN_VARIANCE_THRESHOLD, N_TOP_GENES)
    X_processed = X_filtered_t.T  # samples × genes

    # Persist
    X_processed.to_parquet(processed_dir / "rnaseq_processed.parquet")
    y.to_frame().to_parquet(processed_dir / "labels.parquet")
    pd.Series(gene_list, name="gene_id").to_csv(processed_dir / "gene_list.csv", index=False)

    logger.info("Processing complete. Shape: %s, Labels: %s", X_processed.shape, y.value_counts().to_dict())
    return X_processed, y, gene_list


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire & preprocess TCGA multi-omics data")
    parser.add_argument("--project", default=TCGA_PROJECT_ID)
    parser.add_argument("--max-files", type=int, default=None, help="Limit downloads for testing")
    args = parser.parse_args()
    run_acquisition(args.project, args.max_files)
