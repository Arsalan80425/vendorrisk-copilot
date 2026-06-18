# Architecture

VendorRisk Copilot is organized around a linear data path from raw operational sources to scored, evidence-backed vendor decisions. Each layer has a single responsibility and writes artifacts that downstream services can consume without re-running upstream jobs.

## System Overview

```text
Raw CSVs + contracts
        |
        v
Data quality validation -> feature pipeline -> vendor_features.csv
        |                         |
        |                         v
        |                 LogisticRegression + RandomForest + MLflow
        |
        v
Contract chunking -> FAISS index under artifacts/faiss/
        |
        v
LangGraph workflow -> FastAPI /analyze-vendor -> Streamlit + n8n payload
```

The design prioritizes **deterministic, local-first AI**: scoring, explanations, and recommendations are reproducible without paid API calls. Contract retrieval uses local embeddings; optional LLM providers are reserved for future extension via environment configuration.

---

## Data Sources

All raw inputs live under `data/`.

| Source | Path | Contents |
| --- | --- | --- |
| Vendor master | `data/raw/vendor_master.csv` | Vendor profile, category, country, criticality, compliance status, contract dates, owners, annual contract value |
| Invoices | `data/raw/invoices.csv` | Invoice dates, amounts, payment status, PO match flags, currency, cost center |
| Support tickets | `data/raw/support_tickets.csv` | Severity, resolution hours, SLA hours, status, issue type |
| Contracts | `data/contracts/*.txt` | Plain-text MSA clauses for payment, SLA, compliance, renewal, termination, security, and escalation |

The synthetic generator (`src.data_generation.generate_synthetic_data`) creates a deterministic demo corpus with intentional high-risk patterns (duplicate pending invoices, SLA breaches, missing compliance, near-term renewals).

---

## Configuration Layer

**Module:** `src/config.py`

Centralizes project paths and environment loading:

- `DATA_DIR`, `RAW_DATA_DIR`, `CONTRACTS_DIR`, `PROCESSED_DATA_DIR`
- `ARTIFACTS_DIR`, `MODEL_DIR`, `FAISS_DIR`, `REPORTS_DIR`, `MLRUNS_DIR`
- `VENDOR_FEATURES_PATH`, `MODEL_PATH`
- `RISK_LEVEL_THRESHOLDS` for business-friendly risk bands

`ensure_directories()` creates output folders before pipelines run. `.env` is loaded from the project root via `python-dotenv`.

---

## Data Quality Validation

**Module:** `src.data_quality.validate_data`

Runs before or alongside feature engineering to produce a structured `DataQualityReport`:

```json
{
  "passed": true,
  "errors": [],
  "warnings": [],
  "summary": {}
}
```

**Checks include:**

- Required column presence per dataset
- Non-null constraints on key fields
- Allowed values for criticality, compliance status, invoice status, PO match, severity, ticket status
- Valid date ordering (contract start/end, invoice due/paid dates)
- Referential integrity (invoice and ticket `vendor_id` exists in vendor master)
- Duplicate invoice candidates (same vendor, date, amount)
- PO mismatch warnings
- Missing or expired compliance evidence flags

Errors block a strict pipeline run; warnings surface business signals (e.g., duplicate invoice candidates) without failing the demo.

---

## ETL / Feature Pipeline

**Module:** `src.pipelines.build_features`

Aggregates raw tables into **one row per vendor** and writes `data/processed/vendor_features.csv`.

**Representative features:**

| Feature | Meaning |
| --- | --- |
| `total_spend`, `paid_spend` | Invoice amount aggregates |
| `pending_invoice_amount`, `overdue_invoice_amount` | Open AP exposure |
| `invoice_count`, `duplicate_invoice_count`, `po_mismatch_count` | Invoice integrity signals |
| `ticket_count`, `critical_ticket_count` | Support volume |
| `average_resolution_hours`, `max_resolution_hours` | Responsiveness |
| `sla_breach_count`, `sla_breach_rate` | Contractual performance |
| `days_until_contract_end`, `renewal_within_90_days` | Renewal urgency |
| `compliance_missing_flag`, `criticality_score` | Governance signals |
| `high_risk_vendor`, `risk_score_rule` | Rule-based label and score for training and dashboard KPIs |

The pipeline calls validation internally and prints a summary so operators see data quality status at build time.

---

## ML Risk Model

**Modules:** `src/ml/train_model.py`, `src/ml/predict_risk.py`, `src/ml/monitor_drift.py`

### Training

1. Load or build `vendor_features.csv`
2. Train **LogisticRegression** (with `StandardScaler`) and **RandomForestClassifier**
3. Evaluate accuracy, precision, recall, F1, and ROC-AUC on a held-out split
4. Log each run to **MLflow** under `mlruns/`
5. Persist the best model by F1 to `artifacts/model/vendor_risk_model.joblib`

**Additional artifacts:**

- `artifacts/model/feature_names.json` — model input columns
- `artifacts/model/training_baseline.json` — feature distribution baseline for drift
- `artifacts/model/model_results.json` — comparison table of all models

### Prediction

`predict_vendor(vendor_id)` returns:

- Probability-like `risk_score` (0–1)
- `risk_level`: High (≥0.75), Medium (≥0.45), Low otherwise
- Deterministic `top_risk_factors` derived from feature thresholds
- `estimated_financial_exposure` from duplicate, overdue, pending, and SLA penalty heuristics
- `recommended_action` based on level and factor mix

### Drift Monitoring

`monitor_drift()` compares current features to the training baseline and writes `artifacts/reports/drift_report.json`.

---

## Contract RAG

**Modules:** `src/rag/ingest_contracts.py`, `src/rag/retrieve_clauses.py`

### Ingestion

1. Read every `.txt` file in `data/contracts/`
2. Split text into chunks by blank lines and headings
3. Classify clause types: payment, SLA, compliance, renewal, termination
4. Embed chunks with **sentence-transformers** (`all-MiniLM-L6-v2` by default)
5. Build a **FAISS** index and persist metadata

**Artifacts:**

- `artifacts/faiss/contracts.index`
- `artifacts/faiss/chunks.json`
- `artifacts/faiss/metadata.json`

### Retrieval

`retrieve_contract_clauses(vendor_name, query, top_k)`:

- Prefers chunks matching the vendor name
- Falls back to global search when no vendor match exists
- Returns scored clauses with source file, clause type, and text

`generate_contract_evidence_summary(...)` produces a **source-grounded explanation** from retrieved clauses without calling an external LLM.

---

## LangGraph Workflow

**Module:** `src/agents/vendor_workflow.py`

The workflow is a directed graph over `VendorRiskState`:

```text
START
  -> validate_input
  -> load_vendor_features
  -> predict_vendor_risk
  -> retrieve_contract_evidence
  -> generate_explanation
  -> decide_procurement_action
  -> prepare_automation_payload
  -> END
```

| Node | Responsibility |
| --- | --- |
| `validate_input` | Confirm `vendor_id` exists in features |
| `load_vendor_features` | Load the vendor feature row into state |
| `predict_vendor_risk` | Call ML scorer; populate score, level, factors, exposure |
| `retrieve_contract_evidence` | Query RAG using top risk factors as the search query |
| `generate_explanation` | Build deterministic evidence summary |
| `decide_procurement_action` | Map risk level to approve / review / escalate |
| `prepare_automation_payload` | Emit n8n-ready JSON with owners, exposure, and evidence sources |

**Procurement actions:**

- **High** → Escalate to procurement manager; block approval until review
- **Medium** → Request compliance or SLA review before renewal/payment approval
- **Low** → Approve normal processing

The automation payload includes `event_type`, owners, exposure, risk factors, and evidence source file names.

---

## FastAPI Service

**Module:** `src/api/main.py`

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Project metadata and endpoint list |
| `/health` | GET | Service health plus artifact readiness flags |
| `/vendors` | GET | Vendor list for dashboards |
| `/vendor-risk-summary` | GET | Portfolio KPIs and exposure totals |
| `/analyze-vendor` | POST | Full LangGraph analysis for one vendor |
| `/docs` | GET | OpenAPI Swagger UI |

**Design choices:**

- Startup checks artifact presence without auto training (fail fast with helpful CLI hints)
- Pydantic request/response models in `src/utils/schemas.py`
- CORS enabled for local dashboard development
- Errors return safe messages without leaking stack traces to clients

---

## Streamlit Dashboard

**Module:** `dashboard/app.py`

**Sections:**

1. **Executive Summary** — vendor counts, spend, pending exposure, duplicate exposure, SLA breaches, renewals, total estimated exposure
2. **Vendor Risk Explorer** — select vendor, run analysis via FastAPI or local workflow fallback
3. **BI Charts** — spend by category, risk distribution, SLA breaches, pending invoices, compliance status, renewal timeline, duplicate counts
4. **High-Risk Review Queue** — vendors flagged for escalation
5. **ROI Simulation** — duplicate, pending/overdue, SLA penalty, renewal exposure, and total preventable exposure

The dashboard reads `API_BASE_URL` from the environment. When FastAPI is unavailable, it falls back to local files and `analyze_vendor_with_workflow`.

---

## n8n Automation

**File:** `n8n/vendor_risk_workflow.example.json`

Self-hosted **n8n Community Edition** workflow for RPA-style procurement automation:

1. Manual trigger with sample `vendor_id` (default `V001`)
2. HTTP POST to `/analyze-vendor`
3. IF branch on `risk_level == High`
4. High-risk path → Slack alert with vendor name, score, exposure, and recommended action
5. High-risk path → append review row to Google Sheets (vendor ID, name, score, level, action, exposure, timestamp)

Configure FastAPI URL, Slack webhook, and Google Sheets credentials in n8n. Import and setup steps are in `docs/n8n_setup.md`.

---

## MLflow Tracking

**Storage:** `mlruns/`

Each training run logs:

- Model type and hyperparameters
- Metrics (accuracy, precision, recall, F1, ROC-AUC)
- Model artifact path

The docker-compose stack includes an MLflow UI service on port 5000 with the local `mlruns` volume mounted.

---

## Deployment

| File | Role |
| --- | --- |
| `Dockerfile.api` | Python 3.11 slim, Uvicorn on port 8000 |
| `Dockerfile.dashboard` | Python 3.11 slim, Streamlit on port 8501 |
| `docker-compose.yml` | API, dashboard, MLflow with shared volumes for `data`, `artifacts`, `mlruns` |

Containers run the serving layer only. Operators generate artifacts on the host before `docker compose up --build`.

---

## Output Artifact Map

| Artifact | Producer | Consumer |
| --- | --- | --- |
| `data/processed/vendor_features.csv` | `build_features` | ML, API, dashboard |
| `artifacts/model/vendor_risk_model.joblib` | `train_model` | `predict_risk`, workflow |
| `artifacts/faiss/contracts.index` | `ingest_contracts` | `retrieve_clauses`, workflow |
| `artifacts/reports/drift_report.json` | `monitor_drift` | Ops / model review |
| `mlruns/` | `train_model` | MLflow UI |
| Automation payload JSON | LangGraph workflow | n8n, webhooks, ticketing |

---

## Deterministic AI Design

No paid LLM is required for the reference implementation:

- **Risk factors** come from feature thresholds
- **Recommendations** come from rule templates keyed on risk level
- **Contract explanations** summarize retrieved clause text
- **Embeddings** run locally via sentence-transformers

This makes demos reproducible, cost-free, and suitable for portfolio and interview walkthroughs. Optional LLM integration can be added behind `LLM_PROVIDER` without changing the core workflow shape.
