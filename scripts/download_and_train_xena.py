#!/usr/bin/env python3
"""Download Xena datasets until sample targets (based on GDC counts) are met, then train models.

Usage:
    python scripts/download_and_train_xena.py --projects TCGA-LUAD TCGA-PRAD
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import GDC_ENDPOINT, XENA_ENDPOINT, RAW_DIR
from src.data.xena_acquisition import list_datasets, download_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def gdc_count_for_project(project_id: str) -> int:
    url = f"{GDC_ENDPOINT}/files"
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
        ],
    }
    params = {"filters": json.dumps(filters), "size": "1", "format": "JSON"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]["pagination"]["total"]


def safe_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "_").replace(" ", "_")


def download_until_target(cohort: str, target_samples: int, hub: str = "tcgaHub") -> List[str]:
    out_dir = Path(RAW_DIR) / "xena" / cohort.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = list_datasets(cohort, hub)
    expr_candidates = [ds for ds in datasets if any(x in ds.get("name", "") for x in ["HiSeqV2", "RSEM", "FPKM", "RSEM_normalized"]) ]
    logger.info("Found %d expression candidate datasets for cohort %s", len(expr_candidates), cohort)

    downloaded = []
    total_samples = 0
    for ds in expr_candidates:
        ds_name = ds.get("name")
        if not ds_name:
            continue
        save_path = out_dir / f"{safe_name(ds_name)}.tsv.gz"
        if save_path.exists():
            try:
                df = pd.read_csv(save_path, sep="\t", index_col=0, nrows=1)
                samples = df.shape[1]
            except Exception:
                samples = 0
            total_samples += samples
            downloaded.append(ds_name)
            if total_samples >= target_samples:
                break
            continue

        logger.info("Downloading Xena dataset %s", ds_name)
        df = download_matrix(ds_name, hub)
        if df is None:
            logger.warning("Skipping %s (failed to download)", ds_name)
            continue

        # Persist as gzipped TSV
        try:
            content = df.to_csv(sep="\t").encode("utf-8")
            (out_dir / f"{safe_name(ds_name)}.tsv.gz").write_bytes(content)
        except Exception:
            pass

        samples = df.shape[1]
        total_samples += samples
        downloaded.append(ds_name)
        logger.info("Downloaded %s (%d samples). Total accumulated: %d/%d", ds_name, samples, total_samples, target_samples)

        if total_samples >= target_samples:
            break

    logger.info("Finished downloads for %s: %d datasets, %d samples", cohort, len(downloaded), total_samples)
    return downloaded


def run_training_for_dataset(dataset_id: str):
    cmd = [
        sys.executable, "run_pipeline.py", "--mode", "train", "--real-data",
        "--data-source", "xena", "--xena-dataset", dataset_id
    ]
    logger.info("Running training: %s", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+", required=True)
    parser.add_argument("--hub", default="tcgaHub")
    args = parser.parse_args()

    for proj in args.projects:
        # Convert TCGA-PRAD -> TCGA.PRAD for Xena cohort naming
        cohort = proj.replace("-", ".")
        logger.info("Computing target for %s", proj)
        try:
            target = gdc_count_for_project(proj)
        except Exception as e:
            logger.warning("GDC query failed for %s: %s — defaulting to 500", proj, e)
            target = 500

        logger.info("Target samples for %s: %d", proj, target)
        downloaded = download_until_target(cohort, target, hub=args.hub)
        if not downloaded:
            logger.warning("No datasets downloaded for %s — skipping training", proj)
            continue

        # Choose the largest downloaded dataset by sample count (attempt to read)
        best = None
        best_samples = 0
        for ds in downloaded:
            try:
                # read small preview to count columns
                df = download_matrix(ds, args.hub)
                if df is None:
                    continue
                if df.shape[1] > best_samples:
                    best_samples = df.shape[1]
                    best = ds
            except Exception:
                continue

        if not best:
            best = downloaded[0]

        logger.info("Starting training on best dataset for %s: %s (%d samples)", proj, best, best_samples)
        run_training_for_dataset(best)


if __name__ == "__main__":
    main()
