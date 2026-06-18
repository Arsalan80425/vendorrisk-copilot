"""Build vendor-level risk features from raw procurement data."""

from __future__ import annotations

from datetime import date
import re

import pandas as pd

from src.config import RAW_DATA_DIR, VENDOR_FEATURES_PATH, ensure_directories
from src.data_quality.validate_data import validate_all

CRITICALITY_SCORE = {"Low": 1, "Medium": 2, "High": 3}


def _slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _high_risk_label(row: pd.Series) -> int:
    """Flag high-risk vendors when at least two major risk criteria are present."""
    criteria = [
        row["compliance_missing_flag"] == 1,
        row["duplicate_invoice_count"] > 0,
        row["sla_breach_count"] >= 2,
        row["renewal_within_90_days"] == 1,
        row["pending_invoice_amount"] > 100000,
        row["po_mismatch_count"] >= 2,
    ]
    return int(sum(criteria) >= 2)


def _risk_score(row: pd.Series) -> float:
    """Create a deterministic score for ranking and dashboard use."""
    score = (
        row["compliance_missing_flag"] * 20
        + min(row["duplicate_invoice_count"], 3) * 10
        + min(row["sla_breach_count"], 4) * 8
        + row["renewal_within_90_days"] * 12
        + min(row["pending_invoice_amount"] / 100000, 2) * 10
        + min(row["po_mismatch_count"], 3) * 8
        + max(0, row["criticality_score"] - 1) * 4
        + min(row["annual_contract_value"] / 1_000_000, 1.5) * 12
    )
    return round(min(float(score), 100.0), 2)


def build_features(reference_date: date | None = None) -> pd.DataFrame:
    """Create one feature row per vendor and save it to data/processed."""
    ensure_directories()
    report = validate_all()
    if not report.passed:
        messages = [issue.message for issue in report.errors]
        raise ValueError(f"Cannot build features due to data quality errors: {messages}")

    reference_date = reference_date or date.today()
    vendors = pd.read_csv(RAW_DATA_DIR / "vendor_master.csv")
    invoices = pd.read_csv(RAW_DATA_DIR / "invoices.csv")
    tickets = pd.read_csv(RAW_DATA_DIR / "support_tickets.csv")

    vendors["annual_spend"] = vendors["annual_contract_value"]
    vendors["owner"] = vendors["business_owner"]
    vendors["contract_file"] = vendors["vendor_name"].apply(lambda name: f"{_slugify(str(name))}_contract.txt")
    vendors["security_rating"] = vendors["compliance_status"].map(
        {
            "Complete": 88,
            "Incomplete": 72,
            "Missing SOC2": 64,
            "Expired Insurance": 67,
            "Missing DPA": 66,
        }
    ).fillna(70)

    invoices["duplicate_invoice_flag"] = invoices.duplicated(
        subset=["vendor_id", "invoice_date", "amount"], keep=False
    )
    invoices["paid_amount"] = invoices["amount"].where(invoices["invoice_status"].eq("Paid"), 0)
    invoices["pending_amount"] = invoices["amount"].where(invoices["invoice_status"].eq("Pending"), 0)
    invoices["overdue_amount"] = invoices["amount"].where(invoices["invoice_status"].eq("Overdue"), 0)
    invoices["po_mismatch_flag"] = invoices["po_match"].eq("No")

    invoice_features = invoices.groupby("vendor_id", as_index=False).agg(
        total_spend=("amount", "sum"),
        paid_spend=("paid_amount", "sum"),
        pending_invoice_amount=("pending_amount", "sum"),
        overdue_invoice_amount=("overdue_amount", "sum"),
        invoice_count=("invoice_id", "count"),
        duplicate_invoice_count=("duplicate_invoice_flag", "sum"),
        po_mismatch_count=("po_mismatch_flag", "sum"),
    )

    tickets["sla_breach_flag"] = tickets["resolution_hours"] > tickets["sla_hours"]
    tickets["critical_ticket_flag"] = tickets["severity"].eq("Critical")
    ticket_features = tickets.groupby("vendor_id", as_index=False).agg(
        average_resolution_hours=("resolution_hours", "mean"),
        max_resolution_hours=("resolution_hours", "max"),
        ticket_count=("ticket_id", "count"),
        critical_ticket_count=("critical_ticket_flag", "sum"),
        sla_breach_count=("sla_breach_flag", "sum"),
        open_ticket_count=("status", lambda values: values.isin(["Open", "Escalated"]).sum()),
    )

    features = vendors.merge(invoice_features, on="vendor_id", how="left").merge(
        ticket_features, on="vendor_id", how="left"
    )
    numeric_cols = [
        "total_spend",
        "paid_spend",
        "pending_invoice_amount",
        "overdue_invoice_amount",
        "invoice_count",
        "duplicate_invoice_count",
        "po_mismatch_count",
        "average_resolution_hours",
        "max_resolution_hours",
        "ticket_count",
        "critical_ticket_count",
        "sla_breach_count",
        "open_ticket_count",
    ]
    features[numeric_cols] = features[numeric_cols].fillna(0)
    features["sla_breach_rate"] = (
        features["sla_breach_count"] / features["ticket_count"].replace(0, pd.NA)
    ).fillna(0)
    features["contract_end_date"] = pd.to_datetime(features["contract_end"])
    features["days_until_contract_end"] = (
        features["contract_end_date"] - pd.Timestamp(reference_date)
    ).dt.days
    features["renewal_within_90_days"] = features["days_until_contract_end"].between(0, 90).astype(int)
    features["compliance_missing_flag"] = features["compliance_status"].ne("Complete").astype(int)
    features["criticality_score"] = features["criticality"].map(CRITICALITY_SCORE).fillna(1).astype(int)
    features["high_risk_vendor"] = features.apply(_high_risk_label, axis=1)

    # Compatibility aliases for the API, model, and dashboard layers.
    features["pending_invoice_exposure"] = features["pending_invoice_amount"]
    features["days_to_renewal"] = features["days_until_contract_end"]
    features["missing_compliance"] = features["compliance_missing_flag"].astype(bool)
    features["avg_response_hours"] = features["average_resolution_hours"]
    features["max_days_late"] = 0
    features["late_invoice_count"] = features["overdue_invoice_amount"].gt(0).astype(int)
    features["total_invoice_amount"] = features["total_spend"]
    features["risk_score_rule"] = features.apply(_risk_score, axis=1)
    features["risk_level_rule"] = features["risk_score_rule"].apply(_risk_level)
    features["risk_target"] = features["high_risk_vendor"]

    features.to_csv(VENDOR_FEATURES_PATH, index=False)
    return features


def main() -> None:
    """Validate inputs, build features, and print pipeline outputs."""
    report = validate_all()
    print("Validation summary:")
    print(report.model_dump_json(indent=2))
    if not report.passed:
        raise SystemExit(1)

    features = build_features()
    print(f"Saved {len(features)} vendor feature rows to {VENDOR_FEATURES_PATH}")


if __name__ == "__main__":
    main()
