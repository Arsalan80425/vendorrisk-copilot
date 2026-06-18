# 3-Minute Demo Script — VendorRisk Copilot

**Audience:** Hiring manager, procurement stakeholder, or AI Solutions Engineering interviewer  
**Duration:** ~3 minutes  
**Prerequisites:** Artifacts built, API and dashboard running (or dashboard in local fallback mode)

```bash
python -m src.pipelines.build_features
python -m src.rag.ingest_contracts
python -m src.ml.train_model
uvicorn src.api.main:app --reload
streamlit run dashboard/app.py
```

---

## Script

### 0:00 — Open dashboard `[Step 1]`

> "I'll start in the VendorRisk Copilot dashboard — this is the procurement team's single view of vendor risk across invoices, support tickets, contracts, and ML scoring."

Open `http://127.0.0.1:8501`. Confirm the FastAPI connection caption or note local fallback mode.

---

### 0:15 — Explain the business problem `[Step 2]`

> "Procurement teams usually see invoices in AP, SLA issues in support, and contract terms in legal — but not together at the vendor level. That creates blind spots: duplicate invoices, missing compliance before renewal, SLA breaches without contract context, and unpaid exposure on high-risk vendors."

---

### 0:35 — Show KPIs `[Step 3]`

Point to the **Executive Summary** row:

> "At portfolio level we see total vendors, high-risk count, total spend, pending invoice exposure, duplicate invoice exposure, SLA breach count, contracts expiring in 90 days, and total estimated financial exposure — all computed from the same feature pipeline the ML model uses."

Highlight one number that stands out (e.g., duplicate exposure or contracts expiring within 90 days).

---

### 0:55 — Analyze high-risk vendor DataBridge `[Step 4]`

In **Vendor Risk Explorer**, select:

**`V001 — DataBridge Solutions`**

Click **Analyze Vendor**.

> "DataBridge is our flagship high-risk vendor — missing SOC2, duplicate pending invoices, SLA breaches, and a contract renewal coming up in about 30 days."

---

### 1:15 — Show risk factors `[Step 5]`

Point to the analysis panel metrics and **Top risk factors**:

> "The workflow combines ML scoring with deterministic factor extraction — duplicate invoice candidates, SLA breach events, compliance gaps, renewal urgency, and high pending AP. These aren't black-box — they're tied to actual feature values."

Read 2–3 factors aloud.

---

### 1:30 — Show contract evidence `[Step 6]`

Scroll to **Contract evidence** and **Source-grounded explanation**:

> "This is local RAG over contract text — no paid LLM. We chunk MSAs, embed them with sentence-transformers, store them in FAISS, and retrieve clauses that match the risk factors. Procurement sees payment terms, SLA obligations, compliance requirements, and renewal notice language with source files."

Click one row in the evidence table if time allows.

---

### 1:50 — Show recommendation `[Step 7]`

Point to **Recommendation** and **Human review**:

> "For high-risk vendors the workflow recommends escalation — block approval until procurement and the business owner review. Medium risk triggers compliance or SLA review before renewal or payment approval."

---

### 2:05 — Show n8n workflow `[Step 8]`

Switch to n8n (or show `n8n/vendor_risk_workflow.example.json` / screenshot).

> "The API returns an automation payload designed for n8n. The workflow POSTs to `/analyze-vendor`, branches on high risk, and routes a Slack notification to the business owner with exposure, factors, and evidence sources. This is how insight becomes action without manual copy-paste."

Trace: **Manual Trigger → Set Vendor → HTTP Request → IF High Risk → Slack**.

---

### 2:25 — Show MLflow run `[Step 9]`

Open `http://127.0.0.1:5000` (or `mlflow ui` locally).

> "Training compares LogisticRegression and RandomForest, logs metrics to MLflow, and persists the best model by F1. That gives us experiment history and a path to drift monitoring — not just a pickle file on someone's laptop."

Show the latest runs with F1 and accuracy.

---

### 2:45 — Close with ROI and AI Solutions Engineering relevance `[Step 10]`

Return to the dashboard **ROI Simulation** section (or reference [docs/roi_case_study.md](roi_case_study.md)):

> "The ROI panel breaks exposure into duplicate invoices, pending/overdue AP, SLA penalties, and renewal risk — about $910K in preventable exposure in our 12-vendor pilot narrative. VendorRisk Copilot is an AI Solutions Engineering project because it spans data quality, ETL, ML with MLflow, local RAG, LangGraph orchestration, FastAPI, BI, Docker, and automation — with business-readable outputs, not just model metrics."

> "Happy to walk through the API, the LangGraph graph, or the architecture doc next."

---

## Timing Cheat Sheet

| Time | Step | Focus |
| --- | --- | --- |
| 0:00 | Dashboard | Visual anchor |
| 0:15 | Problem | Why this exists |
| 0:35 | KPIs | Portfolio signal |
| 0:55 | DataBridge | High-risk drill-down |
| 1:15 | Risk factors | Explainability |
| 1:30 | Contract evidence | RAG grounding |
| 1:50 | Recommendation | Action |
| 2:05 | n8n | Automation handoff |
| 2:25 | MLflow | MLOps credibility |
| 2:45 | ROI + close | Business + role fit |

---

## Backup Talking Points

If something fails live:

| Issue | Fallback |
| --- | --- |
| FastAPI down | Dashboard uses local LangGraph fallback — say so explicitly |
| MLflow empty | Show `artifacts/model/model_results.json` |
| RAG index missing | Run `python -m src.rag.ingest_contracts` |
| Analysis slow | Pre-run `curl -X POST .../analyze-vendor -d '{"vendor_id":"V001"}'` |

## Optional 30-Second API Add-On

```bash
curl -X POST http://127.0.0.1:8000/analyze-vendor \
  -H "Content-Type: application/json" \
  -d "{\"vendor_id\":\"V001\"}"
```

> "The same workflow is exposed as a production-shaped API with Pydantic models and OpenAPI docs at `/docs`."
