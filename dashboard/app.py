"""Streamlit BI dashboard for VendorRisk Copilot."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.vendor_workflow import analyze_vendor_with_workflow
from src.config import VENDOR_FEATURES_PATH
from src.ml.predict_risk import estimate_financial_exposure, risk_level
from src.pipelines.build_features import build_features

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="VendorRisk Copilot",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_vendor_features() -> pd.DataFrame:
    """Load processed features, building them if the processed file is missing."""
    if VENDOR_FEATURES_PATH.exists():
        return pd.read_csv(VENDOR_FEATURES_PATH)
    try:
        return build_features()
    except Exception as exc:
        st.error(
            "Processed features are unavailable. Run `python -m src.pipelines.build_features` "
            f"from the project root. Details: {exc}"
        )
        return pd.DataFrame()


def api_get(path: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Call local FastAPI if it is running; otherwise return None."""
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def api_analyze_vendor(vendor_id: str) -> dict[str, Any] | None:
    """Call local FastAPI analysis endpoint when available."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze-vendor",
            json={"vendor_id": vendor_id},
            timeout=12,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def local_risk_summary(features: pd.DataFrame) -> dict[str, float | int]:
    """Compute dashboard summary metrics from local feature rows."""
    risk_levels = features["risk_score_rule"].apply(lambda value: risk_level(float(value) / 100))
    exposures = [
        estimate_financial_exposure(row, str(level))
        for (_, row), level in zip(features.iterrows(), risk_levels)
    ]
    duplicate_exposure = (
        features["pending_invoice_amount"]
        * features["duplicate_invoice_count"].clip(upper=3)
        / features["invoice_count"].replace(0, 1)
    ).sum()
    return {
        "total_vendors": int(len(features)),
        "high_risk_vendors": int(risk_levels.eq("High").sum()),
        "medium_risk_vendors": int(risk_levels.eq("Medium").sum()),
        "low_risk_vendors": int(risk_levels.eq("Low").sum()),
        "total_spend": round(float(features["total_spend"].sum()), 2),
        "pending_invoice_exposure": round(float(features["pending_invoice_amount"].sum()), 2),
        "duplicate_invoice_exposure": round(float(duplicate_exposure), 2),
        "sla_breach_count": int(features["sla_breach_count"].sum()),
        "contracts_expiring_90_days": int(features["renewal_within_90_days"].sum()),
        "total_estimated_exposure": round(float(sum(exposures)), 2),
    }


def local_analyze_vendor(vendor_id: str) -> dict[str, Any]:
    """Run local LangGraph analysis when the FastAPI service is unavailable."""
    return analyze_vendor_with_workflow(vendor_id)


def money(value: float | int | None) -> str:
    """Format currency for dashboard metrics."""
    return f"${float(value or 0):,.0f}"


def with_risk_level(features: pd.DataFrame) -> pd.DataFrame:
    """Attach dashboard risk levels based on probability-style thresholds."""
    enriched = features.copy()
    enriched["dashboard_risk_level"] = enriched["risk_score_rule"].apply(lambda value: risk_level(float(value) / 100))
    return enriched


features = load_vendor_features()
st.title("VendorRisk Copilot — Procurement Risk Intelligence Dashboard")

api_health = api_get("/health")
if api_health:
    st.caption("Connected to local FastAPI backend.")
else:
    st.caption("FastAPI backend not detected. Dashboard is using local files and local workflow functions.")

if features.empty:
    st.stop()

features = with_risk_level(features)
api_summary = api_get("/vendor-risk-summary")
summary = local_risk_summary(features)
if isinstance(api_summary, dict):
    summary.update(api_summary)
    summary["pending_invoice_exposure"] = round(float(features["pending_invoice_amount"].sum()), 2)

st.header("Executive Summary")
kpi_row_1 = st.columns(4)
kpi_row_1[0].metric("Total vendors", int(summary["total_vendors"]))
kpi_row_1[1].metric("High-risk vendors", int(summary["high_risk_vendors"]))
kpi_row_1[2].metric("Total spend", money(summary["total_spend"]))
kpi_row_1[3].metric("Pending invoice exposure", money(summary["pending_invoice_exposure"]))

kpi_row_2 = st.columns(4)
kpi_row_2[0].metric("Duplicate invoice exposure", money(summary["duplicate_invoice_exposure"]))
kpi_row_2[1].metric("SLA breach count", int(summary["sla_breach_count"]))
kpi_row_2[2].metric("Contracts expiring within 90 days", int(summary["contracts_expiring_90_days"]))
kpi_row_2[3].metric("Estimated financial exposure", money(summary["total_estimated_exposure"]))

st.header("Vendor Risk Explorer")
vendor_options = (
    features["vendor_id"].astype(str) + " — " + features["vendor_name"].astype(str)
).tolist()
selected_vendor = st.selectbox("Select vendor", vendor_options)
selected_vendor_id = selected_vendor.split(" — ")[0]

if st.button("Analyze Vendor", type="primary"):
    analysis = api_analyze_vendor(selected_vendor_id)
    source = "FastAPI" if analysis else "local workflow"
    if analysis is None:
        analysis = local_analyze_vendor(selected_vendor_id)

    st.subheader(f"Analysis from {source}")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Risk score", f"{float(analysis['risk_score']):.2f}")
    metric_cols[1].metric("Risk level", analysis["risk_level"])
    metric_cols[2].metric("Financial exposure", money(analysis["estimated_financial_exposure"]))
    metric_cols[3].metric("Human review", "Required" if analysis["human_review_required"] else "Not required")

    st.write("Top risk factors")
    st.write(analysis["top_risk_factors"])
    st.write("Recommendation")
    st.info(analysis["recommended_action"])
    st.write("Source-grounded explanation")
    st.write(analysis.get("explanation") or "No explanation generated.")

    st.write("Contract evidence")
    evidence = pd.DataFrame(analysis["retrieved_contract_evidence"])
    st.dataframe(evidence, use_container_width=True, hide_index=True)

    st.write("Automation payload")
    st.json(analysis["automation_payload"])

st.header("BI Charts")
chart_col_1, chart_col_2 = st.columns(2)
with chart_col_1:
    spend_by_category = features.groupby("category", as_index=False)["total_spend"].sum()
    st.plotly_chart(
        px.bar(spend_by_category, x="category", y="total_spend", title="Spend by Vendor Category"),
        use_container_width=True,
    )
with chart_col_2:
    risk_distribution = features["dashboard_risk_level"].value_counts().reset_index()
    risk_distribution.columns = ["risk_level", "vendor_count"]
    st.plotly_chart(
        px.pie(risk_distribution, names="risk_level", values="vendor_count", title="Risk Level Distribution"),
        use_container_width=True,
    )

chart_col_3, chart_col_4 = st.columns(2)
with chart_col_3:
    sla_chart = features.sort_values("sla_breach_count", ascending=False).head(15)
    st.plotly_chart(
        px.bar(sla_chart, x="vendor_name", y="sla_breach_count", title="SLA Breaches by Vendor"),
        use_container_width=True,
    )
with chart_col_4:
    pending_chart = features.sort_values("pending_invoice_amount", ascending=False).head(15)
    st.plotly_chart(
        px.bar(pending_chart, x="vendor_name", y="pending_invoice_amount", title="Pending Invoice Exposure by Vendor"),
        use_container_width=True,
    )

chart_col_5, chart_col_6 = st.columns(2)
with chart_col_5:
    compliance_distribution = features["compliance_status"].value_counts().reset_index()
    compliance_distribution.columns = ["compliance_status", "vendor_count"]
    st.plotly_chart(
        px.bar(
            compliance_distribution,
            x="compliance_status",
            y="vendor_count",
            title="Compliance Status Distribution",
        ),
        use_container_width=True,
    )
with chart_col_6:
    renewal_timeline = features.sort_values("days_until_contract_end")
    st.plotly_chart(
        px.scatter(
            renewal_timeline,
            x="contract_end",
            y="vendor_name",
            color="dashboard_risk_level",
            size="annual_contract_value",
            title="Contract Renewal Timeline",
        ),
        use_container_width=True,
    )

duplicate_chart = features.sort_values("duplicate_invoice_count", ascending=False).head(15)
st.plotly_chart(
    px.bar(
        duplicate_chart,
        x="vendor_name",
        y="duplicate_invoice_count",
        title="Duplicate Invoice Count by Vendor",
    ),
    use_container_width=True,
)

st.header("High-Risk Review Queue")
queue = features.loc[
    (features["high_risk_vendor"].eq(1)) | (features["dashboard_risk_level"].eq("High"))
].copy()
queue["recommended_action"] = "Escalate to procurement manager and block approval until review"
queue_cols = [
    "vendor_id",
    "vendor_name",
    "risk_score_rule",
    "high_risk_vendor",
    "pending_invoice_amount",
    "sla_breach_count",
    "duplicate_invoice_count",
    "days_until_contract_end",
    "recommended_action",
]
st.dataframe(
    queue[queue_cols].sort_values(["risk_score_rule", "pending_invoice_amount"], ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.header("ROI Simulation")
duplicate_invoice_exposure = float(summary["duplicate_invoice_exposure"])
pending_overdue_exposure = float(features["pending_invoice_amount"].sum() + features["overdue_invoice_amount"].sum())
sla_penalty_estimate = float((features["sla_breach_count"] * features["annual_contract_value"] * 0.01).sum())
renewal_risk_exposure = float(
    (features["renewal_within_90_days"] * features["annual_contract_value"] * 0.04).sum()
)
preventable_exposure = (
    duplicate_invoice_exposure
    + pending_overdue_exposure
    + sla_penalty_estimate
    + renewal_risk_exposure
)

formula_cols = st.columns(5)
formula_cols[0].metric("Duplicate invoice exposure", money(duplicate_invoice_exposure))
formula_cols[1].metric("Pending/overdue exposure", money(pending_overdue_exposure))
formula_cols[2].metric("SLA penalty estimate", money(sla_penalty_estimate))
formula_cols[3].metric("Renewal risk exposure", money(renewal_risk_exposure))
formula_cols[4].metric("Total preventable exposure", money(preventable_exposure))

st.code(
    "\n".join(
        [
            "duplicate invoice exposure = sum(pending_invoice_amount * duplicate_invoice_count / invoice_count)",
            "pending/overdue exposure = sum(pending_invoice_amount + overdue_invoice_amount)",
            "SLA penalty estimate = sum(sla_breach_count * annual_contract_value * 1%)",
            "renewal risk exposure = sum(renewal_within_90_days * annual_contract_value * 4%)",
            "total estimated preventable exposure = duplicate + pending/overdue + SLA + renewal",
        ]
    ),
    language="text",
)
