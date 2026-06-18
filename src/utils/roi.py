"""ROI helpers for vendor-risk automation scenarios."""

from __future__ import annotations


def estimate_financial_exposure(
    pending_invoice_exposure: float,
    annual_spend: float,
    sla_breach_count: int,
    duplicate_invoice_count: int,
    days_to_renewal: int,
) -> float:
    """Estimate exposure from invoices, SLA issues, duplicate payments, and renewal risk."""
    pending_risk = pending_invoice_exposure * 0.35
    sla_risk = sla_breach_count * min(annual_spend * 0.01, 25000)
    duplicate_risk = duplicate_invoice_count * min(annual_spend * 0.015, 20000)
    renewal_risk = annual_spend * 0.04 if 0 <= days_to_renewal <= 90 else 0
    return round(pending_risk + sla_risk + duplicate_risk + renewal_risk, 2)


def simulate_roi(
    annual_vendor_spend: float,
    analyst_hours_saved_per_month: float,
    hourly_rate: float,
    avoided_exposure: float,
    automation_cost: float,
) -> dict[str, float]:
    """Return a simple annual ROI model for portfolio storytelling."""
    labor_savings = analyst_hours_saved_per_month * hourly_rate * 12
    spend_leakage_reduction = annual_vendor_spend * 0.005
    gross_benefit = labor_savings + avoided_exposure + spend_leakage_reduction
    net_benefit = gross_benefit - automation_cost
    roi_percent = (net_benefit / automation_cost * 100) if automation_cost else 0
    return {
        "labor_savings": round(labor_savings, 2),
        "spend_leakage_reduction": round(spend_leakage_reduction, 2),
        "avoided_exposure": round(avoided_exposure, 2),
        "gross_benefit": round(gross_benefit, 2),
        "net_benefit": round(net_benefit, 2),
        "roi_percent": round(roi_percent, 1),
    }
