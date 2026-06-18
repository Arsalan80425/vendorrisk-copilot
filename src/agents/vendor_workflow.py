"""LangGraph workflow for end-to-end vendor risk analysis."""

from __future__ import annotations

import argparse
import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.ml.predict_risk import load_features, predict_vendor
from src.rag.retrieve_clauses import generate_contract_evidence_summary, retrieve_contract_clauses
from src.utils.schemas import AutomationPayload, ContractEvidence, VendorAnalysisResponse


class VendorRiskState(TypedDict):
    """State passed through the vendor risk LangGraph workflow."""

    vendor_id: str
    vendor_name: str | None
    validation_report: dict | None
    features: dict | None
    risk_score: float | None
    risk_level: str | None
    top_risk_factors: list[str]
    retrieved_contract_evidence: list[dict]
    explanation: str | None
    recommended_action: str | None
    estimated_financial_exposure: float | None
    human_review_required: bool
    automation_payload: dict | None
    errors: list[str]


def _initial_state(vendor_id: str) -> VendorRiskState:
    return {
        "vendor_id": vendor_id,
        "vendor_name": None,
        "validation_report": None,
        "features": None,
        "risk_score": None,
        "risk_level": None,
        "top_risk_factors": [],
        "retrieved_contract_evidence": [],
        "explanation": None,
        "recommended_action": None,
        "estimated_financial_exposure": None,
        "human_review_required": False,
        "automation_payload": None,
        "errors": [],
    }


def validate_input(state: VendorRiskState) -> VendorRiskState:
    """Confirm the requested vendor exists in vendor_features.csv."""
    features = load_features()
    exists = bool(features["vendor_id"].eq(state["vendor_id"]).any())
    state["validation_report"] = {
        "vendor_id": state["vendor_id"],
        "exists": exists,
        "vendor_count": int(len(features)),
    }
    if not exists:
        state["errors"].append(f"Unknown vendor_id: {state['vendor_id']}")
    return state


def load_vendor_features(state: VendorRiskState) -> VendorRiskState:
    """Load the vendor feature row into workflow state."""
    if state["errors"]:
        return state
    features = load_features()
    row = features.loc[features["vendor_id"].eq(state["vendor_id"])].iloc[0]
    state["vendor_name"] = str(row["vendor_name"])
    state["features"] = row.where(row.notna(), None).to_dict()
    return state


def predict_vendor_risk(state: VendorRiskState) -> VendorRiskState:
    """Call the ML risk scorer and write prediction outputs into state."""
    if state["errors"]:
        return state
    try:
        prediction = predict_vendor(state["vendor_id"])
        state["risk_score"] = float(prediction["risk_score"])
        state["risk_level"] = str(prediction["risk_level"])
        state["top_risk_factors"] = list(prediction["top_risk_factors"])
        state["estimated_financial_exposure"] = float(prediction["estimated_financial_exposure"])
    except Exception as exc:
        state["errors"].append(f"Risk prediction failed: {exc}")
    return state


def retrieve_contract_evidence(state: VendorRiskState) -> VendorRiskState:
    """Retrieve contract clauses relevant to the vendor risk factors."""
    if state["risk_score"] is None:
        return state
    try:
        query = " ".join(state["top_risk_factors"]) or "vendor risk contract obligations"
        state["retrieved_contract_evidence"] = retrieve_contract_clauses(
            vendor_name=state["vendor_name"] or state["vendor_id"],
            query=query,
            top_k=5,
        )
    except Exception as exc:
        state["errors"].append(f"Contract retrieval failed: {exc}")
        state["retrieved_contract_evidence"] = []
    return state


def generate_explanation(state: VendorRiskState) -> VendorRiskState:
    """Generate a deterministic source-grounded explanation from retrieved clauses."""
    if state["risk_score"] is None:
        return state
    state["explanation"] = generate_contract_evidence_summary(
        vendor_name=state["vendor_name"] or state["vendor_id"],
        risk_factors=state["top_risk_factors"],
        retrieved_clauses=state["retrieved_contract_evidence"],
    )
    return state


def decide_procurement_action(state: VendorRiskState) -> VendorRiskState:
    """Choose the procurement action from the final risk level."""
    if state["risk_score"] is None:
        return state
    if state["risk_level"] == "High":
        state["recommended_action"] = "Escalate to procurement manager and block approval until review"
        state["human_review_required"] = True
    elif state["risk_level"] == "Medium":
        state["recommended_action"] = "Request compliance or SLA review before renewal/payment approval"
        state["human_review_required"] = True
    else:
        state["recommended_action"] = "Approve normal processing"
        state["human_review_required"] = False
    return state


def prepare_automation_payload(state: VendorRiskState) -> VendorRiskState:
    """Create a workflow payload suitable for n8n or another external orchestrator."""
    if state["risk_score"] is None:
        return state
    features = state["features"] or {}
    state["automation_payload"] = {
        "event_type": "vendor_risk_review_required"
        if state["human_review_required"]
        else "vendor_risk_auto_approved",
        "vendor_id": state["vendor_id"],
        "vendor_name": state["vendor_name"],
        "risk_score": state["risk_score"],
        "risk_level": state["risk_level"],
        "recommended_action": state["recommended_action"],
        "human_review_required": state["human_review_required"],
        "business_owner": features.get("business_owner") or features.get("owner"),
        "account_manager": features.get("account_manager"),
        "estimated_financial_exposure": state["estimated_financial_exposure"],
        "top_risk_factors": state["top_risk_factors"],
        "evidence_sources": sorted(
            {str(clause.get("source")) for clause in state["retrieved_contract_evidence"]}
        ),
    }
    return state


def _build_graph():
    graph = StateGraph(VendorRiskState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("load_vendor_features", load_vendor_features)
    graph.add_node("predict_vendor_risk", predict_vendor_risk)
    graph.add_node("retrieve_contract_evidence", retrieve_contract_evidence)
    graph.add_node("generate_explanation", generate_explanation)
    graph.add_node("decide_procurement_action", decide_procurement_action)
    graph.add_node("prepare_automation_payload", prepare_automation_payload)

    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "load_vendor_features")
    graph.add_edge("load_vendor_features", "predict_vendor_risk")
    graph.add_edge("predict_vendor_risk", "retrieve_contract_evidence")
    graph.add_edge("retrieve_contract_evidence", "generate_explanation")
    graph.add_edge("generate_explanation", "decide_procurement_action")
    graph.add_edge("decide_procurement_action", "prepare_automation_payload")
    graph.add_edge("prepare_automation_payload", END)
    return graph.compile()


def _public_result(state: VendorRiskState) -> dict[str, Any]:
    return {
        "vendor_id": state["vendor_id"],
        "vendor_name": state["vendor_name"],
        "risk_score": state["risk_score"],
        "risk_level": state["risk_level"],
        "top_risk_factors": state["top_risk_factors"],
        "retrieved_contract_evidence": state["retrieved_contract_evidence"],
        "explanation": state["explanation"],
        "recommended_action": state["recommended_action"],
        "estimated_financial_exposure": state["estimated_financial_exposure"],
        "human_review_required": state["human_review_required"],
        "automation_payload": state["automation_payload"],
        "errors": state["errors"],
    }


def analyze_vendor_with_workflow(vendor_id: str) -> dict[str, Any]:
    """Run the LangGraph vendor risk workflow and return a plain dict."""
    result = _build_graph().invoke(_initial_state(vendor_id))
    return _public_result(result)


def run_vendor_analysis(vendor_id: str) -> VendorAnalysisResponse:
    """Compatibility wrapper returning the FastAPI response model."""
    result = analyze_vendor_with_workflow(vendor_id)
    if result["errors"]:
        raise KeyError("; ".join(result["errors"]))

    automation = result["automation_payload"] or {}
    automation_payload = AutomationPayload(
        event_type=str(automation.get("event_type", "vendor_risk_review_required")),
        vendor_id=str(result["vendor_id"]),
        vendor_name=str(result["vendor_name"]),
        risk_level=str(result["risk_level"]),
        recommended_action=str(result["recommended_action"]),
        owner=str(automation.get("business_owner") or automation.get("owner") or ""),
        estimated_financial_exposure=float(result["estimated_financial_exposure"] or 0),
        top_risk_factors=list(result["top_risk_factors"]),
    )
    evidence = [
        ContractEvidence(
            text=str(clause.get("text", "")),
            vendor_name=str(clause.get("vendor_name", result["vendor_name"])),
            clause_type=str(clause.get("clause_type", "general")),
            source=str(clause.get("source", "")),
            score=float(clause.get("score", 0.0)),
        )
        for clause in result["retrieved_contract_evidence"]
    ]
    return VendorAnalysisResponse(
        vendor_id=str(result["vendor_id"]),
        vendor_name=str(result["vendor_name"]),
        risk_score=float(result["risk_score"] or 0),
        risk_level=str(result["risk_level"]),
        top_risk_factors=list(result["top_risk_factors"]),
        retrieved_contract_evidence=evidence,
        explanation=result["explanation"],
        recommended_action=str(result["recommended_action"]),
        estimated_financial_exposure=float(result["estimated_financial_exposure"] or 0),
        human_review_required=bool(result["human_review_required"]),
        automation_payload=automation_payload,
    )


def main() -> None:
    """CLI entrypoint for the vendor risk workflow."""
    parser = argparse.ArgumentParser(description="Run the VendorRisk Copilot LangGraph workflow.")
    parser.add_argument("--vendor-id", required=True, help="Vendor ID to analyze, such as V002.")
    args = parser.parse_args()
    print(json.dumps(analyze_vendor_with_workflow(args.vendor_id), indent=2))


if __name__ == "__main__":
    main()
