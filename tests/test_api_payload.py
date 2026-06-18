from fastapi.testclient import TestClient

from src.api.main import app


def test_analyze_vendor_payload_shape():
    client = TestClient(app)

    response = client.post("/analyze-vendor", json={"vendor_id": "V002"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["vendor_id"] == "V002"
    assert "risk_score" in payload
    assert "retrieved_contract_evidence" in payload
    assert "explanation" in payload
    assert "human_review_required" in payload
    assert payload["automation_payload"]["event_type"] in {
        "vendor_risk_review_required",
        "vendor_risk_auto_approved",
    }
    assert {"text", "vendor_name", "clause_type", "source", "score"}.issubset(
        payload["retrieved_contract_evidence"][0].keys()
    )


def test_health_and_summary_endpoints():
    client = TestClient(app)

    health = client.get("/health").json()
    summary = client.get("/vendor-risk-summary").json()

    assert health["status"] == "ok"
    assert {"model_exists", "rag_index_exists", "features_exists"}.issubset(health.keys())
    assert summary["total_vendors"] >= 1
    assert "total_estimated_exposure" in summary
