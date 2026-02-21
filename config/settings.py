"""
Centralised configuration for the Multi-Omics Disease Risk Pipeline.

All sensitive values (API keys, emails) are read exclusively from environment
variables so they are never hard-coded.  Non-sensitive defaults are defined here
and can be overridden via environment variables as well.

Load a .env file before importing this module (e.g. via python-dotenv in the
entry-points) to inject secrets without modifying this file.
"""
from __future__ import annotations

import os
from pathlib import Path

# ─── Project root & data paths ────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = BASE_DIR / "models_artifacts"
RESULTS_DIR   = BASE_DIR / "results"

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# ─── TCGA / GDC ───────────────────────────────────────────────────────────────
TCGA_PROJECT_ID: str = os.getenv("TCGA_PROJECT_ID", "TCGA-BRCA")
GDC_ENDPOINT:    str = "https://api.gdc.cancer.gov"

# ─── Xena (UCSC) – Alternative data source (2200+ datasets) ──────────────────
XENA_ENDPOINT:   str = "https://xenabrowser.net/api"
XENA_HUB:        str = os.getenv("XENA_HUB", "tcgaHub")  # tcgaHub | pancanHub | publicHub
DATA_SOURCE:     str = os.getenv("DATA_SOURCE", "gdc")   # gdc | xena

# ─── Data preprocessing ───────────────────────────────────────────────────────
# Number of top high-variance genes to retain after filtering
N_TOP_GENES: int = int(os.getenv("N_TOP_GENES", "500"))
# Genes whose per-sample variance is below this threshold are dropped first
MIN_VARIANCE_THRESHOLD: float = float(os.getenv("MIN_VARIANCE_THRESHOLD", "0.1"))

# Stratified split ratios (must sum to 1.0)
TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15

# ─── Cross-validation ─────────────────────────────────────────────────────────
CV_FOLDS: int = int(os.getenv("CV_FOLDS", "5"))

# ─── SHAP biomarker extraction ────────────────────────────────────────────────
# How many top genes to record per CV fold
TOP_GENES_PER_FOLD: int = int(os.getenv("TOP_GENES_PER_FOLD", "30"))
# Minimum number of folds a gene must appear in to be "stable"
MIN_FOLD_APPEARANCES: int = int(os.getenv("MIN_FOLD_APPEARANCES", "3"))

# ─── MLflow experiment tracking ───────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:///mlruns")

# ─── PubMed / NCBI Entrez ─────────────────────────────────────────────────────
# Required for literature retrieval; use a real address to comply with NCBI ToS
ENTREZ_EMAIL:         str = os.getenv("ENTREZ_EMAIL", "user@example.com")
# Optional — increases NCBI rate limit from 3 req/s to 10 req/s
ENTREZ_API_KEY:       str = os.getenv("ENTREZ_API_KEY", "")
# Maximum PubMed abstracts fetched per gene
PUBMED_MAX_ABSTRACTS: int = int(os.getenv("PUBMED_MAX_ABSTRACTS", "5"))

# ─── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_API_KEY:     str   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL:       str   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_AUDIT_MODEL: str   = os.getenv("GROQ_AUDIT_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.0"))
GROQ_MAX_TOKENS:  int   = int(os.getenv("GROQ_MAX_TOKENS", "4096"))

# ─── FastAPI server ───────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
