"""Rule-based vendor risk scoring for lightweight deployments (no joblib/ML)."""

from __future__ import annotations

import pandas as pd

from src.config import VENDOR_FEATURES_PATH


def _risk_level(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def _compute_rule_score(row: pd.Series) -> float:
    score = 0.0
    compliance_missing = int(row.get("compliance_missing_flag", 0)) == 1
    if not compliance_missing:
        compliance_status = str(row.get("compliance_status", "")).lower()
        compliance_missing = "missing" in compliance_status or "expired" in compliance_status
    if compliance_missing:
        score += 0.25
    if int(row.get("duplicate_invoice_count", 0)) > 0:
        score += 0.20
    sla_breaches = int(row.get("sla_breach_count", 0))
    if sla_breaches >= 2:
        score += 0.25
    elif sla_breaches >= 1:
        score += 0.15
    if float(row.get("pending_invoice_amount", 0)) > 100000:
        score += 0.20
    if int(row.get("po_mismatch_count", 0)) >= 2:
        score += 0.15
    if int(row.get("renewal_within_90_days", 0)) == 1:
        score += 0.15
    if float(row.get("overdue_invoice_amount", 0)) > 50000:
        score += 0.15
    return min(score, 1.0)


def _top_risk_factors(row: pd.Series) -> list[str]:
    factors: list[str] = []
    if int(row.get("duplicate_invoice_count", 0)) > 0:
        factors.append(f"{int(row['duplicate_invoice_count'])} duplicate invoice candidates detected")
    if int(row.get("sla_breach_count", 0)) > 0:
        factors.append(f"{int(row['sla_breach_count'])} SLA breach events identified")
    if int(row.get("compliance_missing_flag", 0)) == 1:
        factors.append("Compliance evidence is missing, expired, or incomplete")
    if int(row.get("renewal_within_90_days", 0)) == 1:
        factors.append(f"Contract renewal is within {int(row['days_until_contract_end'])} days")
    if float(row.get("pending_invoice_amount", 0)) > 100000:
        factors.append(f"High pending invoice exposure: ${float(row['pending_invoice_amount']):,.0f}")
    if int(row.get("po_mismatch_count", 0)) >= 2:
        factors.append(f"{int(row['po_mismatch_count'])} invoices have PO mismatches")
    if float(row.get("overdue_invoice_amount", 0)) > 50000:
        factors.append(f"Overdue invoice exposure: ${float(row['overdue_invoice_amount']):,.0f}")
    return factors[:6] or ["No material risk concentration identified"]


def _estimate_financial_exposure(row: pd.Series, level: str) -> float:
    duplicate_exposure = 0.0
    if int(row.get("duplicate_invoice_count", 0)) > 0:
        duplicate_exposure = float(row.get("pending_invoice_amount", 0)) * min(
            int(row["duplicate_invoice_count"]), 3
        ) / max(float(row.get("invoice_count", 1)), 1)

    overdue_exposure = float(row.get("overdue_invoice_amount", 0))
    pending_exposure = float(row.get("pending_invoice_amount", 0)) if level in {"High", "Medium"} else 0.0
    sla_penalty = float(row.get("sla_breach_count", 0)) * max(
        float(row.get("annual_contract_value", row.get("annual_spend", 0))) * 0.01,
        5000,
    )
    return round(duplicate_exposure + overdue_exposure + pending_exposure + sla_penalty, 2)


def _recommended_action(level: str) -> str:
    if level == "High":
        return "Escalate to procurement manager and block approval until review"
    if level == "Medium":
        return "Request compliance or SLA review before renewal/payment approval"
    return "Approve normal processing"


def predict_vendor_risk_lightweight(vendor_id: str) -> dict:
    """Score one vendor using rule-based logic and vendor_features.csv only."""
    if not VENDOR_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Vendor features file not found: {VENDOR_FEATURES_PATH}")

    features = pd.read_csv(VENDOR_FEATURES_PATH)
    matches = features.loc[features["vendor_id"].eq(vendor_id)]
    if matches.empty:
        known = ", ".join(features["vendor_id"].astype(str).tolist())
        raise KeyError(f"Unknown vendor_id '{vendor_id}'. Known vendors: {known}")

    row = matches.iloc[0]
    risk_score = round(_compute_rule_score(row), 4)
    level = _risk_level(risk_score)
    factors = _top_risk_factors(row)
    exposure = _estimate_financial_exposure(row, level)

    return {
        "vendor_id": str(row["vendor_id"]),
        "vendor_name": str(row["vendor_name"]),
        "risk_score": risk_score,
        "risk_level": level,
        "top_risk_factors": factors,
        "recommended_action": _recommended_action(level),
        "estimated_financial_exposure": exposure,
        "human_review_required": level in {"High", "Medium"},
        "features": row.where(row.notna(), None).to_dict(),
    }
