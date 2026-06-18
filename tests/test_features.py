from datetime import date

from src.pipelines.build_features import build_features


def test_build_features_creates_vendor_level_rows():
    features = build_features(reference_date=date(2026, 6, 17))

    assert len(features) == 25
    expected_columns = {
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
        "sla_breach_rate",
        "days_until_contract_end",
        "renewal_within_90_days",
        "compliance_missing_flag",
        "criticality_score",
        "high_risk_vendor",
    }
    assert expected_columns.issubset(features.columns)
    assert features.loc[features["vendor_id"].eq("V001"), "duplicate_invoice_count"].iloc[0] == 2
    assert features.loc[features["vendor_id"].eq("V001"), "po_mismatch_count"].iloc[0] >= 2


def test_high_risk_vendor_label_exists_and_flags_expected_vendor():
    features = build_features(reference_date=date(2026, 6, 17))

    assert "high_risk_vendor" in features.columns
    assert features["high_risk_vendor"].isin([0, 1]).all()
    assert features["high_risk_vendor"].nunique() == 2
    assert features.loc[features["vendor_name"].eq("DataBridge Solutions"), "high_risk_vendor"].iloc[0] == 1
    assert features.loc[features["vendor_name"].eq("Supportly India"), "high_risk_vendor"].iloc[0] == 0
