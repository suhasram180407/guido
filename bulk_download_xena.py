#!/usr/bin/env python
"""
Bulk download Xena datasets.

Usage:
    python bulk_download_xena.py --cancer breast --output data/raw/
    python bulk_download_xena.py --dataset-id TCGA.BRCA.sampleMap/HiSeqV2 --output data/raw/
"""
import argparse
import gzip
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

XENA_API = "https://xenabrowser.net/api"

CANCER_COHORTS = {
    "breast": ["TCGA.BRCA"],
    "lung": ["TCGA.LUAD", "TCGA.LUSC"],
    "kidney": ["TCGA.KIRC", "TCGA.KIRP", "TCGA.KICH"],
    "colon": ["TCGA.COAD"],
    "liver": ["TCGA.LIHC"],
    "pancreas": ["TCGA.PAAD"],
    "prostate": ["TCGA.PRAD"],
    "melanoma": ["TCGA.SKCM"],
    "glioma": ["TCGA.GBM", "TCGA.LGG"],
    "leukemia": ["TCGA.LAML"],
}


def list_datasets(cohort: str, hub: str = "tcgaHub") -> List[dict]:
    """Fetch all datasets for a cohort."""
    url = f"{XENA_API}/dataset"
    try:
        resp = requests.post(url, json={"host": hub, "cohort": cohort}, timeout=30)
        resp.raise_for_status()
        return resp.json() or []
    except Exception as e:
        logger.error(f"Failed to list datasets for {cohort}: {e}")
        return []


def download_dataset(dataset_id: str, output_path: Path, hub: str = "tcgaHub") -> bool:
    """
    Download a single Xena dataset.
    
    Returns True if successful, False otherwise.
    """
    url = f"{XENA_API}/download"
    params = {"host": hub, "dataset": dataset_id}
    
    logger.info(f"Downloading {dataset_id}...")
    try:
        resp = requests.get(url, params=params, timeout=300, stream=True)
        resp.raise_for_status()
        
        # Get total size for progress bar
        total_size = int(resp.headers.get('content-length', 0))
        
        # Download with progress bar
        downloaded = 0
        chunks = []
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dataset_id) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
        
        content = b''.join(chunks)
        
        # Decompress if gzipped
        if content[:2] == b'\x1f\x8b':
            content = gzip.decompress(content)
        
        # Save to file (keep as TSV for raw storage)
        output_path.write_bytes(content)
        logger.info(f"✅ Saved to {output_path}")
        
        # Also try to parse and display stats
        try:
            df = pd.read_csv(BytesIO(content), sep="\t", index_col=0, nrows=100)
            logger.info(f"   Matrix preview: ~{df.shape[0]:,}+ genes × {df.shape[1]} samples")
        except:
            pass
        
        return True
    except Exception as e:
        logger.error(f"❌ Download failed for {dataset_id}: {e}")
        return False


def bulk_download_by_cancer(cancer_type: str, output_dir: Path, hub: str = "tcgaHub", max_per_cohort: int = 5):
    """Download all expression matrices for a cancer type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cohorts = CANCER_COHORTS.get(cancer_type.lower(), [])
    if not cohorts:
        logger.error(f"Unknown cancer type: {cancer_type}")
        logger.info(f"Available: {', '.join(CANCER_COHORTS.keys())}")
        return
    
    total_downloaded = 0
    for cohort in cohorts:
        logger.info(f"\n📊 Getting datasets for {cohort}...")
        datasets = list_datasets(cohort, hub)
        
        # Filter for expression matrices (HiSeqV2, RPPA, CNV, etc)
        expr_datasets = [
            ds for ds in datasets
            if any(x in ds['name'] for x in ['HiSeqV2', 'RPPA', 'cnv', 'mutation', 'miRNA'])
        ][:max_per_cohort]
        
        logger.info(f"   Found {len(expr_datasets)} expression datasets (showing first {max_per_cohort})")
        
        for ds in expr_datasets:
            safe_name = ds['name'].replace('/', '_').replace(' ', '_')
            output_path = output_dir / f"{safe_name}.tsv.gz"
            
            if output_path.exists():
                logger.info(f"⏭️  Skipping {output_path.name} (already downloaded)")
                continue
            
            if download_dataset(ds['name'], output_path, hub):
                total_downloaded += 1
    
    logger.info(f"\n✅ Downloaded {total_downloaded} datasets to {output_dir}")


def bulk_download_by_dataset_id(dataset_ids: List[str], output_dir: Path, hub: str = "tcgaHub"):
    """Download specific datasets by ID."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_id in dataset_ids:
        safe_name = dataset_id.replace('/', '_').replace(' ', '_')
        output_path = output_dir / f"{safe_name}.tsv.gz"
        download_dataset(dataset_id, output_path, hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk download Xena cancer datasets")
    parser.add_argument("--cancer", type=str, help=f"Cancer type: {', '.join(CANCER_COHORTS.keys())}")
    parser.add_argument("--dataset-id", type=str, nargs="+", help="Specific dataset IDs (e.g., TCGA.BRCA.sampleMap/HiSeqV2)")
    parser.add_argument("--output", type=str, default="data/raw/xena", help="Output directory")
    parser.add_argument("--hub", type=str, default="tcgaHub", help="Xena hub (tcgaHub, pancanHub, publicHub)")
    parser.add_argument("--max-per-cohort", type=int, default=5, help="Max datasets per cohort (for cancer mode)")
    
    args = parser.parse_args()
    
    if args.cancer:
        bulk_download_by_cancer(args.cancer, args.output, args.hub, args.max_per_cohort)
    elif args.dataset_id:
        bulk_download_by_dataset_id(args.dataset_id, args.output, args.hub)
    else:
        parser.print_help()
