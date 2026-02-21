# 🧬 Multi-Omics Disease Risk Modeling System

A **production-ready, LLM-guided multi-omics disease risk pipeline**.

- 🧠 Machine Learning performs statistical risk prediction  
- 📚 Groq LLM performs structured, abstract-grounded biological reasoning  
- 🚫 No hallucinations  
- 🚫 No statistical shortcuts  

---

# 📌 Architecture Overview

```text
PIPELINE FLOW

GDC/TCGA (or Synthetic Data)
        │
        ▼
Preprocessing
        │
        ▼
Baselines (LR-L1, RF, XGBoost)
        │
        ▼
Primary Model (ElasticNet / ShallowNN)
        │
        ├── Gate Check (vs Baseline)
        ▼
SHAP Analysis
        │
        ▼
Stable Biomarkers (≥ 3 folds)
        │
        ▼
PubMed Retrieval (Entrez API)
        │
        ▼
Groq LLM
    ├── Evidence Scoring (Step 5)
    └── Adversarial Mode (Step 6)
        │
        ▼
Biomarker Confidence Score (BCS)
= Stability × Evidence × (1 − Vulnerability)
        │
        ▼
Cross-Cohort Validation (ΔAUROC + CI)
        │
        ▼
FastAPI  +  MLflow
Docker Deployment
```

---

# 📁 Project Structure

```text
multi_omics_pipeline/
│
├── config/
│   └── settings.py
│
├── src/
│   ├── data/
│   │   ├── acquisition.py
│   │   ├── splitter.py
│   │   └── synthetic_data.py
│   │
│   ├── models/
│   │   ├── baselines.py
│   │   ├── primary_model.py
│   │   ├── metrics.py
│   │   └── confidence_scoring.py
│   │
│   ├── shap_analysis/
│   │   └── shap_runner.py
│   │
│   ├── literature/
│   │   └── pubmed_retrieval.py
│   │
│   ├── llm/
│   │   └── groq_validator.py
│   │
│   ├── validation/
│   │   └── cross_cohort.py
│   │
│   ├── api/
│   │   ├── app.py
│   │   ├── pipeline.py
│   │   └── schemas.py
│   │
│   └── utils/
│       └── utils.py
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_llm.py
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── run_pipeline.py
└── .env.example
```

---

# 🚀 Quick Start

## 1️⃣ Set Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_...
ENTREZ_EMAIL=your@email.com
ENTREZ_API_KEY=<ncbi_api_key>   # Optional (increases rate limit)
```

---

## 2️⃣ Install Dependencies (Local Development)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## 3️⃣ Run Full Pipeline (Synthetic Data)

```bash
python run_pipeline.py --mode full --synthetic --n-samples 400
```

---

## 4️⃣ Run With Real TCGA Data

```bash
python run_pipeline.py --mode full --real-data --project TCGA-BRCA
```

---

## 5️⃣ Start API Server

```bash
python run_pipeline.py --mode api
```

Or directly:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

📖 API Docs:  
http://localhost:8000/docs

---

# 🔬 Biomedical Audit (10-Stage)

## Run via API

```bash
curl -X POST "http://localhost:8000/audit" \
  -H "Content-Type: application/json" \
  -d @audit_input.json
```

---

## Run via CLI

```bash
python run_pipeline.py \
  --mode audit \
  --project TCGA-BRCA \
  --audit-input audit_input.json
```

Output saved to:

```text
results/<project_id>/biomedical_audit_<disease_name>.json
```

---

# 🐳 Docker Deployment

```bash
docker compose up --build
```

- API → http://localhost:8000  
- MLflow → http://localhost:5000  

⚠ Never bake secrets into Docker images.

---

# 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| POST | `/train` | Full training pipeline |
| POST | `/evaluate` | Evaluate saved model |
| POST | `/biomarkers` | PubMed + LLM scoring + BCS |
| POST | `/literature` | Single-gene validation |
| POST | `/cross-cohort` | External robustness validation |
| POST | `/audit` | 10-stage biomedical AI audit |

All responses return structured JSON.

---

# 🔄 Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `acquisition.py` | GDC → FPKM-UQ → log2(TPM+1) → top 2000 HVGs |
| 2 | `baselines.py` | LR-L1, RF, XGBoost (5-fold CV + MLflow) |
| 3 | `primary_model.py` | ElasticNet / ShallowNN + gate check |
| 4 | `shap_runner.py` | CV-SHAP → stable biomarkers |
| 5 | `pubmed_retrieval.py` | Abstract retrieval |
| 6 | `groq_validator.py` | Evidence scoring + adversarial mode |
| 7 | `confidence_scoring.py` | BCS computation |
| 8 | `cross_cohort.py` | ΔAUROC + bootstrap CI |
| 9 | `app.py` | FastAPI REST interface |
| 10 | Docker | Containerised deployment |

---

# 🛡 LLM Safety Constraints

- LLM receives **only provided PubMed abstracts**
- System prompt forbids prior training knowledge usage
- Strict JSON output (max 3 retries)
- Independent adversarial vulnerability scoring
- LLM never performs predictions

---

# 📊 Biomarker Confidence Score (BCS)

\[
\text{BCS} = S \times \frac{E_{raw} - 1}{4} \times (1 - V)
\]

Where:

- **S** = Stability score ∈ [0,1]  
- **E_raw** = Evidence strength (1–5)  
- **V** = Vulnerability score ∈ [0,1]  

✅ Biomarkers with **BCS ≥ 0.30** are classified as high-confidence.

---

# 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests include:

- Preprocessing
- Data splitting
- Model training
- Metrics
- BCS computation
- LLM JSON validation (mocked)
- API endpoint tests (mocked)

---

# 🔁 Reproducibility

- Fixed random seeds (`fix_seed`)
- Pinned dependencies
- Persisted splits, SHAP outputs, and LLM results
- MLflow experiment tracking

---

# 🔬 Switching Disease Cohorts

Change in `.env`:

```
TCGA_PROJECT_ID=TCGA-LUAD
```

Examples:

- TCGA-BRCA
- TCGA-LUAD
- TCGA-COAD
- TCGA-GBM

Disease keyword is auto-inferred via `get_disease_keyword()`.

---
