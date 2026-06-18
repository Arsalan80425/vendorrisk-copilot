"""Pydantic models shared by API and workflow layers."""

from typing import Any

from pydantic import BaseModel, Field


class VendorAnalysisRequest(BaseModel):
    """Request model for vendor risk analysis."""

    vendor_id: str = Field(..., examples=["V002"], min_length=2)


class ContractEvidence(BaseModel):
    """Retrieved contract clause evidence."""

    text: str
    vendor_name: str | None = None
    clause_type: str
    source: str
    score: float

    @property
    def contract_file(self) -> str:
        """Backward-compatible alias for older callers."""
        return self.source


class AutomationPayload(BaseModel):
    """Payload suitable for webhook-based automation tools such as n8n."""

    event_type: str
    vendor_id: str
    vendor_name: str
    risk_level: str
    recommended_action: str
    owner: str
    estimated_financial_exposure: float
    top_risk_factors: list[str]


class VendorAnalysisResponse(BaseModel):
    """Response returned by the vendor analysis API."""

    vendor_id: str
    vendor_name: str
    risk_score: float
    risk_level: str
    top_risk_factors: list[str]
    retrieved_contract_evidence: list[ContractEvidence]
    explanation: str | None = None
    recommended_action: str
    estimated_financial_exposure: float
    human_review_required: bool = False
    automation_payload: AutomationPayload


class ApiRootResponse(BaseModel):
    """Root endpoint response."""

    project_name: str
    status: str
    available_endpoints: list[str]


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    model_exists: bool
    rag_index_exists: bool
    features_exists: bool
    deployment_mode: str
    lightweight_ready: bool


class VendorListItem(BaseModel):
    """Vendor row returned by /vendors."""

    vendor_id: str
    vendor_name: str
    category: str
    criticality: str


class VendorRiskSummaryResponse(BaseModel):
    """Portfolio risk summary response."""

    total_vendors: int
    high_risk_vendors: int
    medium_risk_vendors: int
    low_risk_vendors: int
    total_spend: float
    total_estimated_exposure: float
    duplicate_invoice_exposure: float
    sla_breach_count: int
    contracts_expiring_90_days: int


class DataQualityIssue(BaseModel):
    """Single data quality finding."""

    dataset: str
    severity: str
    message: str
    field: str | None = None
    affected_rows: int = 0


class DataQualityReport(BaseModel):
    """Validation result for all raw input files."""

    passed: bool
    errors: list[DataQualityIssue]
    warnings: list[DataQualityIssue]
    summary: dict[str, Any]

    @property
    def issues(self) -> list[DataQualityIssue]:
        """Return all findings for backward-compatible callers."""
        return [*self.errors, *self.warnings]


class WorkflowState(BaseModel):
    """Serializable state snapshot for debugging workflow outputs."""

    values: dict[str, Any]
