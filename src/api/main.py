"""FastAPI backend for VendorRisk Copilot."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import src.config as config
from src.config import (
    CONTRACTS_DIR,
    FEATURE_NAMES_PATH,
    INDEX_PATH,
    MODEL_PATH,
    VENDOR_FEATURES_PATH,
    is_lightweight_mode,
)
from src.ml.lightweight_predict import predict_vendor_risk_lightweight
from src.rag.lightweight_retrieve import (
    generate_lightweight_explanation,
    retrieve_contract_clauses_lightweight,
)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

_DEBUG_FILE_PATHS = (
    "data/raw/vendor_master.csv",
    "data/processed/vendor_features.csv",
    "artifacts/model/vendor_risk_model.joblib",
    "artifacts/model/feature_names.json",
    "artifacts/faiss/contracts.index",
    "artifacts/faiss/chunks.json",
)
_DEBUG_ENV_VARS = ("DEPLOYMENT_MODE", "ENABLE_LLM_ON_RENDER", "LLM_PROVIDER")


class DebugFileInfo(BaseModel):
    exists: bool
    size_bytes: int | None = None


class DebugFilesResponse(BaseModel):
    cwd: str
    files: dict[str, DebugFileInfo]
    environment: dict[str, str | None]


def _artifact_status() -> dict[str, bool]:
    return {
        "model_exists": MODEL_PATH.is_file() and FEATURE_NAMES_PATH.is_file(),
        "rag_index_exists": INDEX_PATH.is_file(),
        "features_exists": VENDOR_FEATURES_PATH.is_file(),
    }


def _lightweight_ready() -> bool:
    if not VENDOR_FEATURES_PATH.is_file():
        return False
    if not CONTRACTS_DIR.is_dir():
        return False
    return any(CONTRACTS_DIR.glob("*_contract.txt"))


def _load_features_csv() -> pd.DataFrame:
    if not VENDOR_FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Vendor features file not found: {VENDOR_FEATURES_PATH}")
    return pd.read_csv(VENDOR_FEATURES_PATH)


def _load_features() -> pd.DataFrame:
    if is_lightweight_mode():
        return _load_features_csv()
    from src.ml.predict_risk import load_features

    return load_features()


def _risk_level_for_summary(score: float) -> str:
    if is_lightweight_mode():
        if score >= 0.70:
            return "High"
        if score >= 0.40:
            return "Medium"
        return "Low"
    from src.ml.predict_risk import risk_level

    return risk_level(score)


def _estimate_exposure(row: pd.Series, level: str) -> float:
    if is_lightweight_mode():
        from src.ml.lightweight_predict import _estimate_financial_exposure

        return _estimate_financial_exposure(row, level)
    from src.ml.predict_risk import estimate_financial_exposure

    return estimate_financial_exposure(row, level)


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


def _analyze_vendor_lightweight(vendor_id: str) -> dict[str, Any]:
    logger.info("lightweight analyze-vendor path for vendor_id=%s", vendor_id)
    if not VENDOR_FEATURES_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail="Required artifacts are missing. Run: python -m src.pipelines.build_features",
        )

    prediction_started = time.perf_counter()
    try:
        prediction = predict_vendor_risk_lightweight(vendor_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    logger.info("lightweight prediction completed in %.3fs", time.perf_counter() - prediction_started)

    retrieval_started = time.perf_counter()
    clauses = retrieve_contract_clauses_lightweight(
        vendor_name=prediction["vendor_name"],
        risk_factors=prediction["top_risk_factors"],
        top_k=5,
    )
    logger.info("lightweight contract retrieval completed in %.3fs", time.perf_counter() - retrieval_started)

    explanation = generate_lightweight_explanation(
        vendor_name=prediction["vendor_name"],
        risk_factors=prediction["top_risk_factors"],
        retrieved_clauses=clauses,
    )

    features = prediction.get("features") or {}
    human_review_required = bool(prediction.get("human_review_required"))
    automation_payload = {
        "event_type": "vendor_risk_review_required"
        if human_review_required
        else "vendor_risk_auto_approved",
        "vendor_id": prediction["vendor_id"],
        "vendor_name": prediction["vendor_name"],
        "risk_score": prediction["risk_score"],
        "risk_level": prediction["risk_level"],
        "recommended_action": prediction["recommended_action"],
        "human_review_required": human_review_required,
        "business_owner": features.get("business_owner") or features.get("owner"),
        "account_manager": features.get("account_manager"),
        "estimated_financial_exposure": prediction["estimated_financial_exposure"],
        "top_risk_factors": prediction["top_risk_factors"],
        "evidence_sources": sorted({str(clause.get("source")) for clause in clauses}),
    }

    return {
        "vendor_id": prediction["vendor_id"],
        "vendor_name": prediction["vendor_name"],
        "risk_score": prediction["risk_score"],
        "risk_level": prediction["risk_level"],
        "top_risk_factors": prediction["top_risk_factors"],
        "retrieved_contract_evidence": clauses,
        "explanation": explanation,
        "recommended_action": prediction["recommended_action"],
        "estimated_financial_exposure": prediction["estimated_financial_exposure"],
        "human_review_required": human_review_required,
        "automation_payload": automation_payload,
        "errors": [],
    }


def _analyze_vendor_full(vendor_id: str) -> dict[str, Any]:
    from src.agents.vendor_workflow import (
        _initial_state,
        _public_result,
        decide_procurement_action,
        generate_explanation,
        load_vendor_features,
        predict_vendor_risk,
        prepare_automation_payload,
        retrieve_contract_evidence,
        validate_input,
    )

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

    state = _initial_state(vendor_id)
    state = validate_input(state)
    state = load_vendor_features(state)

    prediction_started = time.perf_counter()
    state = predict_vendor_risk(state)
    logger.info("model prediction completed in %.3fs", time.perf_counter() - prediction_started)

    retrieval_started = time.perf_counter()
    state = retrieve_contract_evidence(state)
    logger.info("contract retrieval completed in %.3fs", time.perf_counter() - retrieval_started)

    explanation_started = time.perf_counter()
    state = generate_explanation(state)
    logger.info("explanation generation completed in %.3fs", time.perf_counter() - explanation_started)

    state = decide_procurement_action(state)
    state = prepare_automation_payload(state)
    return _public_result(state)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Check artifact presence without training or indexing on startup."""
    mode = config.DEPLOYMENT_MODE
    logger.info("Starting VendorRisk Copilot API in deployment_mode=%s", mode)
    app_instance.state.artifact_status = _artifact_status()
    app_instance.state.lightweight_ready = _lightweight_ready()
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
            "GET /debug-files",
            "GET /docs",
        ],
    )


@app.get("/debug-files", response_model=DebugFilesResponse)
def debug_files() -> DebugFilesResponse:
    """Return cwd, artifact file presence, sizes, and non-secret deployment env vars."""
    cwd = Path.cwd()
    files: dict[str, DebugFileInfo] = {}
    for relative_path in _DEBUG_FILE_PATHS:
        path = cwd / relative_path
        if path.is_file():
            files[relative_path] = DebugFileInfo(exists=True, size_bytes=path.stat().st_size)
        else:
            files[relative_path] = DebugFileInfo(exists=False)

    environment = {name: os.environ.get(name) for name in _DEBUG_ENV_VARS}
    return DebugFilesResponse(cwd=str(cwd), files=files, environment=environment)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and artifact readiness."""
    started = time.perf_counter()
    logger.info("health called deployment_mode=%s", config.DEPLOYMENT_MODE)
    status = _artifact_status()
    response = HealthResponse(
        status="ok",
        deployment_mode=config.DEPLOYMENT_MODE,
        lightweight_ready=_lightweight_ready(),
        **status,
    )
    logger.info("health completed in %.3fs", time.perf_counter() - started)
    return response


@app.get("/vendors", response_model=list[VendorListItem])
def list_vendors() -> list[VendorListItem]:
    """Return known vendors for dashboards and demos."""
    try:
        features = _load_features()
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


@app.post("/analyze-vendor", response_model=VendorAnalysisResponse)
def analyze_vendor(request: VendorAnalysisRequest) -> VendorAnalysisResponse:
    """Analyze one vendor through the LangGraph workflow or lightweight fallback."""
    started = time.perf_counter()
    logger.info(
        "analyze-vendor called vendor_id=%s deployment_mode=%s",
        request.vendor_id,
        config.DEPLOYMENT_MODE,
    )

    try:
        if is_lightweight_mode():
            result = _analyze_vendor_lightweight(request.vendor_id)
        else:
            result = _analyze_vendor_full(request.vendor_id)

        if result.get("errors"):
            error_text = "; ".join(result["errors"])
            if "Unknown vendor_id" in error_text:
                raise HTTPException(status_code=404, detail=error_text)
            raise HTTPException(
                status_code=500,
                detail="Vendor analysis failed. Please try again later.",
            )

        response = _workflow_result_to_response(result)
        logger.info("analyze-vendor completed in %.3fs", time.perf_counter() - started)
        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("analyze-vendor failed")
        raise HTTPException(
            status_code=500,
            detail="Vendor analysis failed. Please try again later.",
        ) from None


@app.get("/vendor-risk-summary", response_model=VendorRiskSummaryResponse)
def vendor_risk_summary() -> VendorRiskSummaryResponse:
    """Return portfolio-level vendor risk and exposure summary."""
    try:
        features = _load_features()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Vendor features unavailable. Run: python -m src.pipelines.build_features. Details: {exc}",
        ) from exc

    scored = features.copy()
    if is_lightweight_mode():
        from src.ml.lightweight_predict import _compute_rule_score

        scored["api_risk_level"] = scored.apply(
            lambda row: _risk_level_for_summary(_compute_rule_score(row)),
            axis=1,
        )
    else:
        scored["api_risk_level"] = scored["risk_score_rule"].apply(
            lambda score: _risk_level_for_summary(float(score) / 100)
        )

    exposures = [
        _estimate_exposure(row, str(row["api_risk_level"]))
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
