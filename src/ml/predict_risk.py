"""Prediction and explanation helpers for vendor risk scoring."""

from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, VENDOR_FEATURES_PATH
from src.ml.train_model import FEATURE_NAMES_PATH, train_model
from src.pipelines.build_features import build_features


def load_features() -> pd.DataFrame:
    """Load processed features, building them if needed."""
    if not VENDOR_FEATURES_PATH.exists():
        return build_features()
    return pd.read_csv(VENDOR_FEATURES_PATH)


def load_model_artifacts() -> tuple[Any, list[str], str]:
    """Load the persisted model, feature list, and model name."""
    if not MODEL_PATH.exists() or not FEATURE_NAMES_PATH.exists():
        train_model()
    artifact = joblib.load(MODEL_PATH)
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    return artifact["model"], feature_names, artifact.get("model_name", "unknown")


def risk_level(risk_score: float) -> str:
    """Map a probability-like score to a business-friendly risk level."""
    if risk_score >= 0.75:
        return "High"
    if risk_score >= 0.45:
        return "Medium"
    return "Low"


def _row_to_model_frame(row: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{feature: row.get(feature, 0) for feature in feature_names}]).fillna(0)


def _normalize_decision_score(score: float) -> float:
    return float(1 / (1 + np.exp(-score)))


def predict_risk_score(model: Any, X: pd.DataFrame) -> float:
    """Return a probability-like risk score from predict_proba or decision_function."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
        return round(float(probabilities[positive_index]), 4)
    if hasattr(model, "decision_function"):
        decision_score = model.decision_function(X)
        return round(_normalize_decision_score(float(np.ravel(decision_score)[0])), 4)
    return round(float(model.predict(X)[0]), 4)


def top_risk_factors(row: pd.Series) -> list[str]:
    """Generate deterministic top-factor explanations from feature values."""
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
    if int(row.get("po_mismatch_count", 0)) > 0:
        factors.append(f"{int(row['po_mismatch_count'])} invoices have PO mismatches")
    return factors[:6] or ["No material risk concentration identified"]


def estimate_financial_exposure(row: pd.Series, level: str) -> float:
    """Estimate financial exposure from invoices and SLA penalties."""
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


def recommended_action(level: str, factors: list[str]) -> str:
    """Return a deterministic recommendation for workflow and dashboard use."""
    factor_text = " ".join(factors).lower()
    if level == "High":
        if "compliance" in factor_text:
            return "Open compliance remediation task, hold renewal approval, and request updated audit evidence."
        return "Escalate to procurement owner, review contract protections, and pause discretionary spend."
    if level == "Medium":
        return "Schedule vendor review, validate invoice exposure, and confirm SLA corrective actions."
    return "Continue standard monitoring and refresh vendor evidence at the next review cycle."


def _rule_based_prediction(row: pd.Series) -> dict[str, Any]:
    """Fallback scorer using deterministic rule features when ML artifacts are unavailable."""
    risk_score = round(float(row.get("risk_score_rule", 0)) / 100, 4)
    level = risk_level(risk_score)
    factors = top_risk_factors(row)
    exposure = estimate_financial_exposure(row, level)
    return {
        "vendor_id": str(row["vendor_id"]),
        "vendor_name": str(row["vendor_name"]),
        "model_name": "rule_based",
        "risk_score": risk_score,
        "risk_level": level,
        "top_risk_factors": factors,
        "recommended_action": recommended_action(level, factors),
        "estimated_financial_exposure": exposure,
    }


def predict_vendor(vendor_id: str) -> dict[str, Any]:
    """Score one vendor and return a JSON-serializable risk result."""
    features = load_features()
    matches = features.loc[features["vendor_id"].eq(vendor_id)]
    if matches.empty:
        known = ", ".join(features["vendor_id"].tolist())
        raise KeyError(f"Unknown vendor_id '{vendor_id}'. Known vendors: {known}")

    row = matches.iloc[0]
    try:
        model, feature_names, model_name = load_model_artifacts()
        risk_score = predict_risk_score(model, _row_to_model_frame(row, feature_names))
        level = risk_level(risk_score)
        factors = top_risk_factors(row)
        exposure = estimate_financial_exposure(row, level)
        return {
            "vendor_id": str(row["vendor_id"]),
            "vendor_name": str(row["vendor_name"]),
            "model_name": model_name,
            "risk_score": risk_score,
            "risk_level": level,
            "top_risk_factors": factors,
            "recommended_action": recommended_action(level, factors),
            "estimated_financial_exposure": exposure,
        }
    except Exception:
        return _rule_based_prediction(row)


def score_vendor(vendor_id: str) -> dict[str, object]:
    """Score one vendor and return model output plus feature row context for workflows."""
    features = load_features()
    matches = features.loc[features["vendor_id"].eq(vendor_id)]
    if matches.empty:
        known = ", ".join(features["vendor_id"].tolist())
        raise KeyError(f"Unknown vendor_id '{vendor_id}'. Known vendors: {known}")

    row = matches.iloc[0]
    result = predict_vendor(vendor_id)
    return {
        "row": row,
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "top_risk_factors": result["top_risk_factors"],
        "recommended_action": result["recommended_action"],
        "estimated_financial_exposure": result["estimated_financial_exposure"],
    }


def main() -> None:
    """CLI entrypoint for vendor risk prediction."""
    parser = argparse.ArgumentParser(description="Predict vendor risk from processed features.")
    parser.add_argument("--vendor-id", required=True, help="Vendor ID to score, such as V002.")
    args = parser.parse_args()
    print(json.dumps(predict_vendor(args.vendor_id), indent=2))


if __name__ == "__main__":
    main()
