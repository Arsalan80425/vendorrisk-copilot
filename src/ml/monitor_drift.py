"""Lightweight drift monitoring for vendor risk features."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config import LATEST_DRIFT_REPORT_PATH, VENDOR_FEATURES_PATH, ensure_directories
from src.ml.train_model import TRAINING_BASELINE_PATH, train_model

DRIFT_RULES: dict[str, float] = {
    "pending_invoice_amount": 30.0,
    "sla_breach_rate": 20.0,
    "compliance_missing_flag": 20.0,
    "duplicate_invoice_count": 30.0,
}


def _percent_change(current: float, baseline: float) -> float:
    denominator = abs(baseline) if abs(baseline) > 1e-9 else 1.0
    return round(((current - baseline) / denominator) * 100.0, 2)


def _load_current_features() -> pd.DataFrame:
    if not VENDOR_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Current feature file not found: {VENDOR_FEATURES_PATH}. "
            "Run `python -m src.pipelines.build_features` first."
        )
    return pd.read_csv(VENDOR_FEATURES_PATH)


def monitor_drift() -> dict[str, Any]:
    """Compare current vendor features against the saved training baseline."""
    ensure_directories()
    if not TRAINING_BASELINE_PATH.exists():
        train_model()

    baseline = json.loads(TRAINING_BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_metrics = baseline.get("metrics", {})
    current = _load_current_features()

    metric_rows: list[dict[str, Any]] = []
    for metric_name, threshold_percent in DRIFT_RULES.items():
        if metric_name not in current.columns:
            raise ValueError(f"Current vendor_features.csv is missing drift feature: {metric_name}")
        baseline_value = round(float(baseline_metrics.get(metric_name, 0.0)), 6)
        current_value = round(float(current[metric_name].mean()), 6)
        percent_change = _percent_change(current_value, baseline_value)
        drift_detected = abs(percent_change) > threshold_percent
        metric_rows.append(
            {
                "metric": metric_name,
                "baseline_value": baseline_value,
                "current_value": current_value,
                "percent_change": percent_change,
                "threshold_percent": threshold_percent,
                "drift_detected": drift_detected,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_path": str(TRAINING_BASELINE_PATH),
        "features_path": str(VENDOR_FEATURES_PATH),
        "drift_detected": any(row["drift_detected"] for row in metric_rows),
        "metrics": metric_rows,
    }
    LATEST_DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_report(report: dict[str, Any]) -> None:
    print("Drift monitoring report")
    print(f"Baseline: {report['baseline_path']}")
    print(f"Current:  {report['features_path']}")
    print(f"Overall drift detected: {report['drift_detected']}")
    print()
    print(f"{'metric':<30} {'baseline':>12} {'current':>12} {'% change':>10} {'drift':>8}")
    print("-" * 76)
    for row in report["metrics"]:
        print(
            f"{row['metric']:<30} "
            f"{row['baseline_value']:>12.4f} "
            f"{row['current_value']:>12.4f} "
            f"{row['percent_change']:>9.2f}% "
            f"{str(row['drift_detected']):>8}"
        )
    print()
    print(f"Saved report: {LATEST_DRIFT_REPORT_PATH}")


def main() -> None:
    """CLI entrypoint for drift monitoring."""
    report = monitor_drift()
    _print_report(report)


if __name__ == "__main__":
    main()
