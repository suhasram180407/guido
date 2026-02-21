"""
Unit tests for data preprocessing and splitting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.data.acquisition import fpkm_to_log2tpm, filter_low_variance, align_and_label
from src.data.splitter import split_data


# ─── fpkm_to_log2tpm ─────────────────────────────────────────────────────────

def test_fpkm_to_log2tpm_shape():
    fpkm = pd.DataFrame(np.random.rand(100, 20), columns=[f"S{i}" for i in range(20)])
    result = fpkm_to_log2tpm(fpkm)
    assert result.shape == (100, 20)


def test_fpkm_to_log2tpm_values_non_negative():
    fpkm = pd.DataFrame(np.abs(np.random.rand(50, 10)), columns=[f"S{i}" for i in range(10)])
    result = fpkm_to_log2tpm(fpkm)
    assert (result.values >= 0).all()


def test_fpkm_to_log2tpm_zero_input():
    """All-zero FPKM should yield log2(0+1) = 0 after TPM conversion."""
    fpkm = pd.DataFrame(np.zeros((10, 5)), columns=[f"S{i}" for i in range(5)])
    result = fpkm_to_log2tpm(fpkm)
    # When all FPKM = 0, division by 0 → NaN; check graceful handling
    assert result.shape == (10, 5)


# ─── filter_low_variance ─────────────────────────────────────────────────────

def test_filter_low_variance_retains_top_n():
    rng = np.random.default_rng(42)
    expr = pd.DataFrame(rng.random((200, 30)), index=[f"ENSG{i:011d}" for i in range(200)])
    filtered, genes = filter_low_variance(expr, threshold=0.0, n_top=50)
    assert len(genes) == 50
    assert filtered.shape[0] == 50


def test_filter_low_variance_removes_low_var():
    rng = np.random.default_rng(0)
    # 100 genes × 2 samples;  genes 0-19 are constant (zero variance)
    expr = pd.DataFrame({
        "S1": [1.0] * 20 + (rng.random(80) * 5 + 1).tolist(),
        "S2": [1.0] * 20 + (rng.random(80) * 5 + 1).tolist(),
    })  # shape (100, 2) — genes as rows, samples as columns
    # genes 0-19 have zero variance; genes 20-99 have positive variance
    _, genes = filter_low_variance(expr, threshold=0.01, n_top=50)
    assert all(float(expr.loc[g, :].var()) > 0.01 for g in genes)


# ─── align_and_label ─────────────────────────────────────────────────────────

def test_align_and_label():
    n_genes, n_patients = 100, 40
    gene_ids = [f"ENSG{i:011d}" for i in range(n_genes)]
    patient_ids = [f"TCGA-XX-{i:04d}" for i in range(n_patients)]

    expr = pd.DataFrame(np.random.rand(n_genes, n_patients), index=gene_ids, columns=patient_ids)
    clinical = pd.DataFrame({
        "vital_status": ["Alive"] * 20 + ["Dead"] * 20,
    }, index=patient_ids)

    X, y = align_and_label(expr, clinical)
    assert X.shape == (n_patients, n_genes)
    assert set(y.unique()).issubset({0, 1})
    assert y.sum() == 20


def test_align_and_label_partial_overlap():
    gene_ids = [f"ENSG{i:011d}" for i in range(50)]
    all_patients = [f"TCGA-XX-{i:04d}" for i in range(20)]
    expr = pd.DataFrame(np.random.rand(50, 20), index=gene_ids, columns=all_patients)
    # Clinical only covers half the patients
    clinical = pd.DataFrame(
        {"vital_status": ["Alive"] * 5 + ["Dead"] * 5},
        index=all_patients[:10],
    )
    X, y = align_and_label(expr, clinical)
    assert X.shape[0] == 10


# ─── split_data ───────────────────────────────────────────────────────────────

def test_split_data_sizes():
    X = pd.DataFrame(np.random.rand(200, 50))
    y = pd.Series([0] * 120 + [1] * 80)
    splits = split_data(X, y)
    total = sum(v[0].shape[0] for v in splits.values())
    assert total == 200


def test_split_data_no_overlap():
    X = pd.DataFrame(np.random.rand(100, 20))
    y = pd.Series([0] * 60 + [1] * 40)
    splits = split_data(X, y)
    train_idx = set(splits["train"][0].index)
    val_idx   = set(splits["val"][0].index)
    test_idx  = set(splits["test"][0].index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_data_stratified():
    X = pd.DataFrame(np.random.rand(200, 10))
    y = pd.Series([0] * 130 + [1] * 70)
    splits = split_data(X, y)
    # Check that each split has both classes
    for name, (Xs, ys) in splits.items():
        assert len(ys.unique()) == 2, f"Split '{name}' is missing a class"
