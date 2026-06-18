"""Lightweight drift monitoring for vendor risk features."""

from __future__ import annotations

import json
from typing import Any

from src.config import REPORTS_DIR, ensure_directories
from src.ml.predict_risk import load_features
from src.ml.train_model import TRAINING_BASELINE_PATH, train_model

DRIFT_RULES = {
    "pending_invoice_amount": 0.30,
    "sla_breach_rate": 0.20,
    "compliance_missing_flag": 0.20,
    "duplicate_invoice_count": 0.30,
}


def _relative_change(current: float, baseline: float) -> float:
    denominator = abs(baseline) if abs(baseline) > 1e-9 else 1.0
    return abs(current - baseline) / denominator


def monitor_drift() -> dict[str, Any]:
    """Compare current vendor features against the saved training baseline."""
    ensure_directories()
    if not TRAINING_BASELINE_PATH.exists():
        train_model()

    baseline = json.loads(TRAINING_BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_metrics = baseline.get("metrics", {})
    current = load_features()
    drift_checks: dict[str, dict[str, float | bool]] = {}

    for feature_name, threshold in DRIFT_RULES.items():
        if feature_name not in current.columns:
            raise ValueError(f"Current vendor_features.csv is missing drift feature: {feature_name}")
        baseline_value = float(baseline_metrics.get(feature_name, 0.0))
        current_value = float(current[feature_name].mean())
        relative_change = _relative_change(current_value, baseline_value)
        drift_checks[feature_name] = {
            "baseline_mean": round(baseline_value, 6),
            "current_mean": round(current_value, 6),
            "relative_change": round(relative_change, 6),
            "threshold": threshold,
            "drifted": relative_change > threshold,
        }

    report = {
        "baseline_path": str(TRAINING_BASELINE_PATH),
        "drift_detected": any(check["drifted"] for check in drift_checks.values()),
        "checks": drift_checks,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """CLI entrypoint for drift monitoring."""
    print(json.dumps(monitor_drift(), indent=2))


if __name__ == "__main__":
    main()
