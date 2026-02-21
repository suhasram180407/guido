# UI Integration Guide

This document describes how the frontend (`ui/`) should call the backend API exposed by the FastAPI app in `src/api/app.py`.

Base API URL
- Local dev (default): `http://localhost:8000`

Environment variables (server):
- `UI_ALLOWED_ORIGINS` — comma-separated list of allowed origins for CORS. Defaults to `http://localhost:5173,http://localhost:3000,http://localhost:8000`.
- `GROQ_API_KEY` — required for full Groq LLM-backed biomarker audit endpoints.

Endpoints

- `POST /api/ui/predict` — accept CSV upload or JSON form payload, returns prediction + features + audit report.
  - Form fields:
    - `project_id` (string) — default `TCGA-LUAD`.
    - `disease_name` (string) — free text disease label.
    - `file` — file upload (CSV). The first row is used as the sample.
    - `json_payload` — optional JSON string (used when `file` is not provided). Either an object (single sample) or an array of objects.
  - Example (file upload):
    ```bash
    curl -F "file=@sample.csv" -F "project_id=TCGA-LUAD" -F "disease_name=lung_adenocarcinoma" \
      http://localhost:8000/api/ui/predict
    ```
  - Example (JSON payload):
    ```bash
    curl -F "json_payload={\"GENE1\": 1.23, \"GENE2\": 0.45}" -F "project_id=TCGA-LUAD" \
      http://localhost:8000/api/ui/predict
    ```

- `POST /api/ui/export_pdf` — accepts JSON body with a `report` key (HTML or markdown string). Returns `pdf_base64` (base64-encoded PDF bytes). Requires `weasyprint` on server.
  - Example:
    ```bash
    curl -H "Content-Type: application/json" -d '{"report":"<h1>Report</h1><p>...")}' \
      http://localhost:8000/api/ui/export_pdf
    ```

Notes & Recommendations
- The UI should prefer `file` uploads for tabular inputs. If using `json_payload`, send a serialized JSON string in the `json_payload` form field.
- Responses are JSON. The `prediction` field contains `label` and `prob`. The `report` field contains the biomedical audit output (or an `error` key on failure).
- For production, ensure `GROQ_API_KEY` is set server-side and do not expose it to the browser.

Next steps for frontend wiring
- Implement `PredictionForm` to POST a file or `json_payload` to `/api/ui/predict` and display `prediction` + `report` in `ReportViewer`.
- Add client-side button to POST `report` JSON to `/api/ui/export_pdf` and download the returned PDF (decode base64).
# UI (Vite + React + Tailwind)

This folder contains a minimal charcoal-themed React UI for the Madhav project built with Vite and Tailwind.

Quick start:

1. cd ui
2. npm install
3. npm run dev

Dev server will start on `http://localhost:5174` by default.

Notes:
- The UI expects a backend API endpoint at `/api/ui/predict` that accepts a CSV `file` field and returns JSON: `{ prediction: {label, prob, model, time}, features: [{name, value}], report: "# markdown" }`.
- PDF download is a placeholder — integrate a server-side PDF generator (wkhtmltopdf, WeasyPrint, or a headless browser) for production.
