import time

from fastapi.testclient import TestClient

import src.config as config
from src.api.main import app
from src.ml.lightweight_predict import predict_vendor_risk_lightweight
from src.rag.lightweight_retrieve import retrieve_contract_clauses_lightweight


def test_predict_vendor_risk_lightweight_v001():
    result = predict_vendor_risk_lightweight("V001")
    assert result["vendor_id"] == "V001"
    assert result["vendor_name"] == "DataBridge Solutions"
    assert result["risk_level"] == "High"
    assert result["risk_score"] >= 0.70
    assert result["top_risk_factors"]
    assert result["recommended_action"]
    assert result["estimated_financial_exposure"] > 0


def test_retrieve_contract_clauses_lightweight():
    clauses = retrieve_contract_clauses_lightweight(
        vendor_name="DataBridge Solutions",
        risk_factors=["Compliance evidence is missing", "SLA breach events identified"],
        top_k=3,
    )
    assert clauses
    assert {"text", "vendor_name", "clause_type", "source", "score"}.issubset(clauses[0].keys())
    assert clauses[0]["source"] == "databridge_solutions_contract.txt"


def test_analyze_vendor_lightweight_api(monkeypatch):
    monkeypatch.setattr(config, "DEPLOYMENT_MODE", "lightweight")

    client = TestClient(app)
    started = time.perf_counter()
    response = client.post("/analyze-vendor", json={"vendor_id": "V001"})
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 5
    payload = response.json()
    assert payload["vendor_id"] == "V001"
    assert payload["risk_level"] == "High"
    assert payload["retrieved_contract_evidence"]
    assert payload["explanation"]
    assert payload["automation_payload"]["event_type"] == "vendor_risk_review_required"


def test_health_lightweight_fields(monkeypatch):
    monkeypatch.setattr(config, "DEPLOYMENT_MODE", "lightweight")

    client = TestClient(app)
    health = client.get("/health").json()

    assert health["status"] == "ok"
    assert health["deployment_mode"] == "lightweight"
    assert health["lightweight_ready"] is True
    assert "model_exists" in health
    assert "features_exists" in health
