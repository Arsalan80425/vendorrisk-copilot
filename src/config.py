"""Central path and environment configuration for VendorRisk Copilot."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "full").strip().lower()


def is_lightweight_mode() -> bool:
    """Return True when the API should avoid heavy ML/RAG dependencies."""
    return DEPLOYMENT_MODE == "lightweight"

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CONTRACTS_DIR = DATA_DIR / "contracts"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
FAISS_DIR = ARTIFACTS_DIR / "faiss"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MLRUNS_DIR = BASE_DIR / "mlruns"

VENDOR_FEATURES_PATH = PROCESSED_DATA_DIR / "vendor_features.csv"
MODEL_PATH = MODEL_DIR / "vendor_risk_model.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
INDEX_PATH = FAISS_DIR / "contracts.index"
LATEST_DRIFT_REPORT_PATH = MODEL_DIR / "latest_drift_report.json"

RISK_LEVEL_THRESHOLDS = {
    "high": 70.0,
    "medium": 40.0,
}


def ensure_directories() -> None:
    """Create project output folders used by pipelines and services."""
    for path in (PROCESSED_DATA_DIR, MODEL_DIR, FAISS_DIR, REPORTS_DIR, MLRUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)
