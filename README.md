# Multi-Omics Disease Risk Modeling System

A production-ready, LLM-guided multi-omics disease risk pipeline. Machine learning performs statistical risk prediction; Groq LLM performs structured, abstract-grounded biological reasoning. No hallucinations, no statistical shortcuts.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIPELINE FLOW                               │
│                                                                     │
│  GDC/TCGA  ──► Preprocess ──► Baselines ──► Primary Model          │
│  (or synthetic)               (LR/RF/XGB)   (ElasticNet/NN)        │
│                                     │              │                │
│                               Gate Check ◄─────────┘                │
│                                     │                               │
│                               SHAP Analysis                         │
│                                     │                               │
│                          Stable Biomarkers (≥3 folds)               │
│                                     │                               │
│                          PubMed Retrieval (Entrez)                  │
│                                     │                               │
│                    ┌────────────────┴───────────────┐               │
│                    │          Groq LLM               │              │
│                    │  Evidence Scoring (Step 5)      │              │
│                    │  Adversarial Mode  (Step 6)     │              │
│                    └────────────────┬───────────────┘               │
│                                     │                               │
│                    Biomarker Confidence Score (BCS)                  │
│                    = Stability × Evidence × (1-Vulnerability)       │
│                                     │                               │
│                    Cross-Cohort Validation (ΔAUROC)                 │
│                                     │                               │
│                          FastAPI  /  MLflow                         │
│                          Docker   /  Results                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
multi_omics_pipeline/
├── config/
│   └── settings.py              # All configuration (paths, seeds, hyperparams)
├── src/
│   ├── data/
│   │   ├── acquisition.py       # Step 1 — GDC download + log2TPM normalisation
│   │   ├── splitter.py          # Stratified 70/15/15 splits
│   │   └── synthetic_data.py    # Offline development data generator
│   ├── models/
│   │   ├── baselines.py         # Step 2 — LR-L1, RF, XGBoost + 5-fold CV
│   │   ├── primary_model.py     # Step 3 — ElasticNet / ShallowNN + gate check
│   │   ├── metrics.py           # AUROC, AUPRC, F1, Accuracy + MLflow logging
│   │   └── confidence_scoring.py # Step 7 — BCS computation
│   ├── shap_analysis/
│   │   └── shap_runner.py       # Step 4 — CV-SHAP + stable biomarker extraction
│   ├── literature/
│   │   └── pubmed_retrieval.py  # Step 5a — Entrez PubMed retrival + caching
│   ├── llm/
│   │   └── groq_validator.py    # Steps 5b+6 — Evidence scoring + adversarial LLM
│   ├── validation/
│   │   └── cross_cohort.py      # Step 8 — External cohort ΔAUROC + bootstrap CI
│   ├── api/
│   │   ├── app.py               # Step 9 — FastAPI endpoints
│   │   ├── pipeline.py          # Orchestration layer
│   │   └── schemas.py           # Pydantic request/response models
│   └── utils/
│       └── utils.py             # Seed fixing, logging, JSON helpers
├── tests/
│   ├── test_data.py             # Unit tests: preprocessing & splitting
│   ├── test_models.py           # Unit tests: models & metrics & BCS
│   └── test_llm.py              # Unit tests: LLM JSON parsing (mocked)
│   └── test_api.py              # Integration tests: API endpoints (mocked)
├── Dockerfile
├── docker-compose.yml           # API + MLflow tracking server
├── requirements.txt
├── pyproject.toml               # pytest + ruff config
├── run_pipeline.py              # CLI entry point
└── .env.example
```

---

## Quick Start

### 1. Set Environment Variables

```bash
cp .env.example .env
# Edit .env:
# GROQ_API_KEY=gsk_...
# ENTREZ_EMAIL=your@email.com
# ENTREZ_API_KEY=<ncbi_api_key>   # optional but increases rate limit
```

### 2. Install Dependencies (local dev)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Full Pipeline (Synthetic Data)

```bash
python run_pipeline.py --mode full --synthetic --n-samples 400
```

### 4. Run With Real TCGA Data

```bash
python run_pipeline.py --mode full --real-data --project TCGA-BRCA
```

### 5. Start API Server

```bash
python run_pipeline.py --mode api
# or directly:
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

---

## Biomedical Audit (10-Stage)

### Run via API (`/audit`)

```bash
curl -X POST "http://localhost:8000/audit" \
	-H "Content-Type: application/json" \
	-d '{
		"project_id": "TCGA-BRCA",
		"disease_name": "breast cancer",
		"multi_omics_features": {
			"genomics_features": ["BRCA1", "TP53"],
			"transcriptomics_features": ["MKI67", "ESR1"],
			"proteomics_features": []
		},
		"selected_biomarkers": ["BRCA1", "TP53"],
		"shap_importance_scores": {"BRCA1": 0.91, "TP53": 0.72},
		"stability_scores": {"BRCA1": 0.82, "TP53": 0.61},
		"model_metrics": {
			"auroc": 0.83,
			"accuracy": 0.76,
			"precision": 0.75,
			"recall": 0.74,
			"f1": 0.74
		},
		"training_sample_size": 180,
		"validation_sample_size": 80
	}'
```

### Run via CLI (`mode=audit`)

Create an input JSON file (for example `audit_input.json`) with the same payload structure as above, then run:

```bash
python run_pipeline.py --mode audit --project TCGA-BRCA --audit-input audit_input.json
```

Output is persisted to:

```text
results/<project_id>/biomedical_audit_<disease_name>.json
```

---

## Docker Deployment

```bash
# Build and start API + MLflow
docker compose up --build

# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

Pass secrets via `.env` file or environment variables — never bake them into the image.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| POST | `/train` | Full training pipeline (data → baselines → primary → SHAP) |
| POST | `/evaluate` | Evaluate saved model on val/test split |
| POST | `/biomarkers` | PubMed + LLM evidence scoring + BCS (requires GROQ_API_KEY) |
| POST | `/literature` | Single-gene literature validation |
| POST | `/cross-cohort` | External cohort robustness validation |
| POST | `/audit` | 10-stage biomedical AI audit with strict JSON output |

All responses are structured JSON.

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `src/data/acquisition.py` | GDC download → FPKM-UQ → log2(TPM+1) → top 2000 HVGs |
| 2 | `src/models/baselines.py` | LR-L1, RF, XGBoost with 5-fold CV + MLflow |
| 3 | `src/models/primary_model.py` | ElasticNet/ShallowNN + gate check vs baseline |
| 4 | `src/shap_analysis/shap_runner.py` | CV-SHAP top-30/fold → stable biomarkers (≥3 folds) |
| 5 | `src/literature/` + `src/llm/` | PubMed abstracts → Groq LLM evidence scoring |
| 6 | `src/llm/groq_validator.py` | Adversarial falsification mode + vulnerability score |
| 7 | `src/models/confidence_scoring.py` | BCS = stability × evidence × (1 − vulnerability) |
| 8 | `src/validation/cross_cohort.py` | ΔAUROC + bootstrap CI + biomarker overlap % |
| 9 | `src/api/app.py` | FastAPI REST interface |
| 10 | `Dockerfile` + `docker-compose.yml` | Containerised deployment + MLflow tracking |

---

## LLM Safety Constraints

- LLM **only** receives provided PubMed abstracts as context.
- System prompt explicitly forbids using prior training knowledge.
- All LLM outputs **must** be valid JSON — invalid responses are retried (max 3×) then rejected.
- Adversarial mode independently searches for weaknesses grounded in the same abstracts.
- LLM **never** makes predictions — only structured biological reasoning.

---

## Biomarker Confidence Score Formula

$$\text{BCS} = S \times \frac{E_{raw} - 1}{4} \times (1 - V)$$

Where:
- $S$ = stability score (fold appearance frequency / CV_FOLDS) ∈ [0, 1]
- $E_{raw}$ = LLM evidence strength (1–5), normalised to [0, 1]
- $V$ = vulnerability score from adversarial mode ∈ [0, 1]

Biomarkers with BCS ≥ 0.30 are classified as **high-confidence**.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: preprocessing, data splitting, model training, metrics, BCS computation, LLM JSON parsing (mocked), and API endpoints (mocked).

---

## Reproducibility

- All random seeds fixed via `fix_seed(RANDOM_SEED)` at startup.
- Dependencies pinned in `requirements.txt`.
- All splits, models, SHAP results, and LLM outputs are persisted to disk.
- MLflow tracks every experiment with hyperparameters and metrics.

---

## Switching Disease Cohorts

Change `TCGA_PROJECT_ID` in `.env` (e.g., `TCGA-LUAD`, `TCGA-COAD`, `TCGA-GBM`). The PubMed disease keyword is auto-inferred from the project ID via `get_disease_keyword()` in `pubmed_retrieval.py`.
#   g u i d o  
 