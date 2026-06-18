# Resume & Social Copy — VendorRisk Copilot

Use these variants on resumes, LinkedIn, GitHub, and portfolio sites. Adjust company names and metrics to match your narrative.

---

## 1-Line Project Bullet

> Built **VendorRisk Copilot**, an end-to-end procurement risk platform using FastAPI, Streamlit, scikit-learn, MLflow, FAISS-based contract RAG, and LangGraph workflows to detect duplicate invoices, SLA breaches, compliance gaps, renewal risk, and estimated financial exposure.

---

## MLOps Bullet (safer wording)

> Implemented local MLOps workflow for vendor-risk modeling, comparing Logistic Regression and Random Forest models, tracking evaluation metrics, saving best-model artifacts, storing feature baselines, and adding lightweight drift monitoring.

---

## 3-Bullet Version

- Built **VendorRisk Copilot**, a procurement risk automation platform integrating data validation, vendor-level feature engineering, and ML scoring (Logistic Regression + Random Forest) with local MLOps training, evaluation metrics, and lightweight drift monitoring.
- Implemented **local contract RAG** with sentence-transformers and FAISS to retrieve payment, SLA, compliance, and renewal clauses, grounding recommendations without paid LLM APIs.
- Delivered **LangGraph orchestration**, FastAPI endpoints, Streamlit BI dashboard, n8n automation payloads, Docker deployment, and ROI simulation estimating ~$910K preventable exposure in a 12-vendor pilot scenario.

---

## 6-Bullet Version

- Designed and built **VendorRisk Copilot**, an AI Solutions Engineering portfolio project that automates vendor risk review across invoices, support tickets, contracts, and vendor master data.
- Implemented **data quality validation** for schema checks, referential integrity, duplicate invoice detection, PO mismatches, and missing compliance evidence with structured error/warning reports.
- Built an **ETL feature pipeline** producing vendor-level signals: spend, pending/overdue exposure, SLA breach rate, renewal urgency, compliance flags, and high-risk labels in `vendor_features.csv`.
- Implemented **local MLOps workflow** for vendor-risk modeling: compared Logistic Regression and Random Forest models, tracked evaluation metrics, saved best-model artifacts, stored feature baselines, and added lightweight **drift monitoring**.
- Developed **contract RAG** (sentence-transformers + FAISS) and a **LangGraph workflow** that chains validation, ML scoring, clause retrieval, source-grounded explanation, procurement actions, and n8n-ready automation payloads.
- Shipped **FastAPI** (`/analyze-vendor`, portfolio summary), **Streamlit BI dashboard** with ROI simulation, **Docker Compose** deployment, and an **n8n example workflow** for high-risk Slack escalation.

---

## LinkedIn Post

**Post:**

I just wrapped **VendorRisk Copilot** — a production-shaped procurement risk platform built for AI Solutions Engineering portfolios.

The problem: procurement teams juggle invoices, SLA tickets, and contracts in separate systems. Duplicate payments, compliance gaps, and renewal risk stay hidden until they're expensive.

What I built:
→ Data validation + vendor-level feature pipeline  
→ ML risk scoring with MLflow tracking (LogisticRegression vs RandomForest)  
→ Local contract RAG (sentence-transformers + FAISS) — no paid LLM required  
→ LangGraph workflow → FastAPI → Streamlit dashboard  
→ n8n automation payloads for high-risk escalation  
→ Docker deployment + ROI simulation

Demo highlight: analyzing **DataBridge Solutions** surfaces duplicate invoice exposure, SLA breaches, missing SOC2, contract renewal urgency, retrieved MSA clauses, and an automation payload for Slack review.

If you're hiring for AI Solutions Engineering, ML platform, or applied AI roles — this project shows the full path from messy operational data to governed features, grounded retrieval, orchestrated workflows, and business-facing BI.

Repo: [link your GitHub URL]  
#AI #MachineLearning #FastAPI #LangGraph #MLOps #Procurement #SolutionsEngineering

---

## GitHub Pinned Repo Description

**Short (≤350 characters):**

End-to-end procurement risk automation: data validation, ETL features, ML scoring with MLflow, FAISS contract RAG, LangGraph workflows, FastAPI, Streamlit BI, n8n payloads, Docker. Detects duplicate invoices, SLA breaches, compliance gaps, and renewal risk with estimated financial exposure.

**Topics/tags:** `fastapi` `streamlit` `scikit-learn` `mlflow` `langgraph` `rag` `faiss` `procurement` `vendor-risk` `docker` `n8n` `ai-solutions-engineering`

**Extended About section:**

VendorRisk Copilot demonstrates how an AI Solutions Engineer moves from raw vendor data to actionable risk signals:

- Validate invoices, tickets, and vendor master data
- Build vendor-level features and train ML risk models
- Retrieve contract clauses with local RAG
- Orchestrate analysis with LangGraph
- Expose results via API, dashboard, and n8n automation

No paid APIs required. Includes Docker Compose, MLflow UI, and ROI case study.

**Quick start:** `pip install -r requirements.txt` → build features → ingest contracts → train model → `uvicorn src.api.main:app --reload` + `streamlit run dashboard/app.py`

---

## Interview Elevator Pitch (30 seconds)

> "VendorRisk Copilot is an end-to-end procurement risk platform I built to show AI Solutions Engineering depth. It ingests vendor, invoice, ticket, and contract data, validates quality, engineers features, trains ML models tracked in MLflow, retrieves contract clauses with local RAG, and runs a LangGraph workflow that outputs risk scores, evidence, recommendations, and n8n automation payloads — all exposed through FastAPI and a Streamlit dashboard with ROI simulation."

---

## Skills Keywords (for ATS)

Python · FastAPI · Streamlit · Pandas · scikit-learn · MLflow · LangGraph · RAG · FAISS · sentence-transformers · Docker · Feature Engineering · Data Quality · MLOps · Procurement · Vendor Risk · API Design · Workflow Automation · n8n · Plotly · Pydantic
