"""Train, compare, and persist vendor risk scoring models."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MLRUNS_DIR, MODEL_DIR, MODEL_PATH, REPORTS_DIR, VENDOR_FEATURES_PATH, ensure_directories
from src.pipelines.build_features import build_features

FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
TRAINING_BASELINE_PATH = MODEL_DIR / "training_baseline.json"
MODEL_RESULTS_PATH = MODEL_DIR / "model_results.json"

RANDOM_STATE = 42
TEST_SIZE = 0.3

FEATURE_COLUMNS = [
    "total_spend",
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
]
TARGET_COLUMN = "high_risk_vendor"


def load_features() -> pd.DataFrame:
    """Load processed vendor features, building them first when needed."""
    if not VENDOR_FEATURES_PATH.exists():
        return build_features()
    return pd.read_csv(VENDOR_FEATURES_PATH)


def _feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"vendor_features.csv is missing required ML feature columns: {missing}")
    return features[FEATURE_COLUMNS].copy().fillna(0)


def _target_series(features: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in features.columns:
        raise ValueError(f"vendor_features.csv is missing target column: {TARGET_COLUMN}")
    target = features[TARGET_COLUMN].astype(int)
    if target.nunique() < 2:
        raise ValueError("Training target must contain both high-risk and non-high-risk vendors.")
    return target


def _split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=None,
        )


def _models() -> dict[str, Any]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def _positive_scores(model: Any, X: pd.DataFrame) -> list[float] | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].tolist()
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        min_score = float(raw_scores.min())
        max_score = float(raw_scores.max())
        if max_score == min_score:
            return [0.5 for _ in raw_scores]
        return ((raw_scores - min_score) / (max_score - min_score)).tolist()
    return None


def _evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float | None], Any]:
    predictions = model.predict(X_test)
    scores = _positive_scores(model, X_test)
    metrics: dict[str, float | None] = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": None,
    }
    if scores is not None and y_test.nunique() > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, scores)), 4)
    return metrics, predictions


def _run_params(model_type: str, train_size: int, test_size: int) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "train_size": train_size,
        "test_size": test_size,
        "feature_count": len(FEATURE_COLUMNS),
        "random_state": RANDOM_STATE,
    }


def _log_to_mlflow(
    model_type: str,
    model: Any,
    metrics: dict[str, float | None],
    params: dict[str, Any],
    y_test: pd.Series,
    predictions: Any,
) -> None:
    """Log model run details to MLflow when the dependency is installed."""
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except ModuleNotFoundError:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "mlflow_skipped.json").write_text(
            json.dumps(
                {
                    "reason": "mlflow is not installed in this Python environment",
                    "install": "pip install -r requirements.txt",
                    "attempted_model": model_type,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("vendorrisk-copilot")
    with mlflow.start_run(run_name=f"vendor-risk-{model_type}"):
        mlflow.log_params(params)
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "feature_names.json").write_text(
                json.dumps(FEATURE_COLUMNS, indent=2),
                encoding="utf-8",
            )
            (tmp_path / "classification_report.txt").write_text(
                classification_report(y_test, predictions, zero_division=0),
                encoding="utf-8",
            )
            (tmp_path / "confusion_matrix.json").write_text(
                json.dumps(confusion_matrix(y_test, predictions).tolist(), indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifacts(str(tmp_path))
        mlflow.sklearn.log_model(model, name="model")


def _model_params(model: Any) -> dict[str, Any]:
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
        return {
            "estimator": estimator.__class__.__name__,
            "class_weight": estimator.class_weight,
            "max_iter": estimator.max_iter,
        }
    return {
        "estimator": model.__class__.__name__,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "class_weight": str(model.class_weight),
        "random_state": model.random_state,
    }


def _training_baseline(features: pd.DataFrame) -> dict[str, Any]:
    baseline_columns = [
        "pending_invoice_amount",
        "sla_breach_rate",
        "compliance_missing_flag",
        "duplicate_invoice_count",
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(features)),
        "metrics": {
            column: round(float(features[column].mean()), 6)
            for column in baseline_columns
        },
    }


def train_model() -> dict[str, Any]:
    """Train LogisticRegression and RandomForest, then persist the best model by F1."""
    ensure_directories()
    features = load_features()
    X = _feature_frame(features)
    y = _target_series(features)
    X_train, X_test, y_train, y_test = _split_data(X, y)
    train_size = len(X_train)
    test_size = len(X_test)

    results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    for model_type, model in _models().items():
        model.fit(X_train, y_train)
        metrics, predictions = _evaluate_model(model, X_test, y_test)
        params = _run_params(model_type, train_size, test_size)
        _log_to_mlflow(model_type, model, metrics, params, y_test, predictions)
        trained_models[model_type] = model
        results.append(
            {
                "model_name": model_type,
                "metrics": metrics,
                "parameters": _model_params(model),
                "run_params": params,
            }
        )

    best_result = max(results, key=lambda item: (item["metrics"]["f1"] or 0, item["metrics"]["accuracy"] or 0))
    best_model_name = best_result["model_name"]
    best_model = trained_models[best_model_name]

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": best_result["metrics"],
        "all_results": results,
    }
    joblib.dump(artifact, MODEL_PATH)
    FEATURE_NAMES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    TRAINING_BASELINE_PATH.write_text(json.dumps(_training_baseline(features), indent=2), encoding="utf-8")
    MODEL_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return {
        "best_model": best_model_name,
        "best_metrics": best_result["metrics"],
        "results": results,
        "model_path": str(MODEL_PATH),
        "feature_names_path": str(FEATURE_NAMES_PATH),
        "training_baseline_path": str(TRAINING_BASELINE_PATH),
        "mlruns_path": str(MLRUNS_DIR),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("Model training results:")
    for result in summary["results"]:
        metrics = result["metrics"]
        print(
            "- {model}: accuracy={accuracy}, precision={precision}, recall={recall}, "
            "f1={f1}, roc_auc={roc_auc}".format(
                model=result["model_name"],
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                roc_auc=metrics["roc_auc"],
            )
        )
    print(f"Best model: {summary['best_model']}")
    print(f"Saved model: {summary['model_path']}")
    print(f"Saved feature list: {summary['feature_names_path']}")
    print(f"Saved training baseline: {summary['training_baseline_path']}")
    print(f"MLflow runs: {summary['mlruns_path']}")


def main() -> None:
    """CLI entrypoint for model training."""
    _print_summary(train_model())


if __name__ == "__main__":
    main()
