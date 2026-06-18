"""FastAPI backend for VendorRisk Copilot."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agents.vendor_workflow import analyze_vendor_with_workflow
from src.config import MODEL_PATH, VENDOR_FEATURES_PATH
from src.ml.predict_risk import estimate_financial_exposure, load_features, risk_level
from src.ml.train_model import FEATURE_NAMES_PATH
from src.rag.ingest_contracts import INDEX_PATH
from src.utils.schemas import (
    ApiRootResponse,
    AutomationPayload,
    ContractEvidence,
    HealthResponse,
    VendorAnalysisRequest,
    VendorAnalysisResponse,
    VendorListItem,
    VendorRiskSummaryResponse,
)

def _artifact_status() -> dict[str, bool]:
    return {
        "model_exists": MODEL_PATH.exists() and FEATURE_NAMES_PATH.exists(),
        "rag_index_exists": INDEX_PATH.exists(),
        "features_exists": VENDOR_FEATURES_PATH.exists(),
    }


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Check artifact presence without training or indexing on startup."""
    app_instance.state.artifact_status = _artifact_status()
    yield


app = FastAPI(
    title="VendorRisk Copilot API",
    version="1.0.0",
    description="Procurement and vendor-risk automation API with ML scoring, contract RAG, and LangGraph workflows.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=ApiRootResponse)
def root() -> ApiRootResponse:
    """Return project metadata and discoverable endpoint paths."""
    return ApiRootResponse(
        project_name="VendorRisk Copilot",
        status="ok",
        available_endpoints=[
            "GET /",
            "GET /health",
            "GET /vendors",
            "GET /vendor-risk-summary",
            "POST /analyze-vendor",
            "GET /docs",
        ],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and artifact readiness."""
    status = _artifact_status()
    return HealthResponse(status="ok", **status)


@app.get("/vendors", response_model=list[VendorListItem])
def list_vendors() -> list[VendorListItem]:
    """Return known vendors for dashboards and demos."""
    try:
        features = load_features()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Vendor features unavailable. Run: python -m src.pipelines.build_features. Details: {exc}",
        ) from exc

    return [
        VendorListItem(
            vendor_id=str(row["vendor_id"]),
            vendor_name=str(row["vendor_name"]),
            category=str(row["category"]),
            criticality=str(row["criticality"]),
        )
        for _, row in features.iterrows()
    ]


def _workflow_result_to_response(result: dict[str, Any]) -> VendorAnalysisResponse:
    automation = result.get("automation_payload") or {}
    evidence = [
        ContractEvidence(
            text=str(clause.get("text", "")),
            vendor_name=str(clause.get("vendor_name", result.get("vendor_name", ""))),
            clause_type=str(clause.get("clause_type", "general")),
            source=str(clause.get("source", "")),
            score=float(clause.get("score", 0.0)),
        )
        for clause in result.get("retrieved_contract_evidence", [])
    ]
    automation_payload = AutomationPayload(
        event_type=str(automation.get("event_type", "vendor_risk_review_required")),
        vendor_id=str(result["vendor_id"]),
        vendor_name=str(result["vendor_name"]),
        risk_level=str(result["risk_level"]),
        recommended_action=str(result["recommended_action"]),
        owner=str(automation.get("business_owner") or automation.get("owner") or ""),
        estimated_financial_exposure=float(result.get("estimated_financial_exposure") or 0),
        top_risk_factors=list(result.get("top_risk_factors", [])),
    )
    return VendorAnalysisResponse(
        vendor_id=str(result["vendor_id"]),
        vendor_name=str(result["vendor_name"]),
        risk_score=float(result["risk_score"]),
        risk_level=str(result["risk_level"]),
        top_risk_factors=list(result.get("top_risk_factors", [])),
        retrieved_contract_evidence=evidence,
        explanation=result.get("explanation"),
        recommended_action=str(result["recommended_action"]),
        estimated_financial_exposure=float(result.get("estimated_financial_exposure") or 0),
        human_review_required=bool(result.get("human_review_required")),
        automation_payload=automation_payload,
    )


@app.post("/analyze-vendor", response_model=VendorAnalysisResponse)
def analyze_vendor(request: VendorAnalysisRequest) -> VendorAnalysisResponse:
    """Analyze one vendor through the LangGraph workflow."""
    status = _artifact_status()
    missing_commands: list[str] = []
    if not status["features_exists"]:
        missing_commands.append("python -m src.pipelines.build_features")
    if not status["model_exists"]:
        missing_commands.append("python -m src.ml.train_model")
    if not status["rag_index_exists"]:
        missing_commands.append("python -m src.rag.ingest_contracts")
    if missing_commands:
        raise HTTPException(
            status_code=500,
            detail=f"Required artifacts are missing. Run: {' && '.join(missing_commands)}",
        )

    try:
        result = analyze_vendor_with_workflow(request.vendor_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vendor analysis failed safely. Details: {exc}") from exc

    if result.get("errors"):
        error_text = "; ".join(result["errors"])
        if "Unknown vendor_id" in error_text:
            raise HTTPException(status_code=404, detail=error_text)
        raise HTTPException(status_code=500, detail=f"Vendor analysis failed safely. Details: {error_text}")
    return _workflow_result_to_response(result)


@app.get("/vendor-risk-summary", response_model=VendorRiskSummaryResponse)
def vendor_risk_summary() -> VendorRiskSummaryResponse:
    """Return portfolio-level vendor risk and exposure summary."""
    try:
        features = load_features()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Vendor features unavailable. Run: python -m src.pipelines.build_features. Details: {exc}",
        ) from exc

    scored = features.copy()
    scored["api_risk_level"] = scored["risk_score_rule"].apply(lambda score: risk_level(float(score) / 100))
    exposures = [
        estimate_financial_exposure(row, str(row["api_risk_level"]))
        for _, row in scored.iterrows()
    ]
    duplicate_invoice_exposure = float(
        (
            scored["pending_invoice_amount"]
            * scored["duplicate_invoice_count"].clip(upper=3)
            / scored["invoice_count"].replace(0, 1)
        ).sum()
    )

    return VendorRiskSummaryResponse(
        total_vendors=int(len(scored)),
        high_risk_vendors=int(scored["api_risk_level"].eq("High").sum()),
        medium_risk_vendors=int(scored["api_risk_level"].eq("Medium").sum()),
        low_risk_vendors=int(scored["api_risk_level"].eq("Low").sum()),
        total_spend=round(float(scored["total_spend"].sum()), 2),
        total_estimated_exposure=round(float(sum(exposures)), 2),
        duplicate_invoice_exposure=round(duplicate_invoice_exposure, 2),
        sla_breach_count=int(scored["sla_breach_count"].sum()),
        contracts_expiring_90_days=int(scored["renewal_within_90_days"].sum()),
    )
