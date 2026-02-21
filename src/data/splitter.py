"""
Data splitting utilities.
Produces stratified train / val / test splits with a fixed random seed.
Optionally loads an external cohort for independent validation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# allow running as script from repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    TCGA_PROJECT_ID,
)

logger = logging.getLogger(__name__)


def load_processed(project_id: str = TCGA_PROJECT_ID) -> Tuple[pd.DataFrame, pd.Series]:
    """Load pre-processed expression matrix and labels from parquet files."""
    proc_dir = PROCESSED_DIR / project_id
    X = pd.read_parquet(proc_dir / "rnaseq_processed.parquet")
    y = pd.read_parquet(proc_dir / "labels.parquet").squeeze("columns")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Stratified three-way split: train / val / test.

    Parameters
    ----------
    X : feature matrix (samples × genes)
    y : binary labels
    train_ratio, val_ratio, test_ratio : must sum to 1.0

    Returns
    -------
    dict with keys "train", "val", "test" each mapping to (X_split, y_split)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # First split: train vs (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        stratify=y,
        random_state=seed,
    )

    # Second split: val vs test (from the remaining pool)
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        stratify=y_tmp,
        random_state=seed,
    )

    splits = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }
    for name, (Xs, ys) in splits.items():
        logger.info("Split %-6s: X=%s  pos_rate=%.3f", name, Xs.shape, ys.mean())
    return splits


def load_external_cohort(cohort_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load an external (independent) cohort for cross-cohort validation.
    Expected files:
        <cohort_path>/rnaseq_processed.parquet
        <cohort_path>/labels.parquet
    """
    p = Path(cohort_path)
    X_ext = pd.read_parquet(p / "rnaseq_processed.parquet")
    y_ext = pd.read_parquet(p / "labels.parquet").squeeze("columns")
    logger.info("External cohort loaded: X=%s", X_ext.shape)
    return X_ext, y_ext
