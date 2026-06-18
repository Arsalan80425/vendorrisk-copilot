# MLOps & Monitoring

VendorRisk Copilot includes a local, runnable MLOps loop: train models with MLflow tracking, persist the champion model, capture a feature baseline, and monitor drift as new vendor features arrive.

## Training Pipeline

From the repository root:

```bash
python -m src.pipelines.build_features
python -m src.ml.train_model
```

`train_model` performs the following steps:

1. Load `data/processed/vendor_features.csv` (or build features if missing).
2. Split features into train/test sets (`test_size=0.3`, `random_state=42`, stratified when possible).
3. Train two classifiers:
   - **LogisticRegression** (with `StandardScaler`)
   - **RandomForestClassifier**
4. Evaluate each model on the held-out test set.
5. Select the **best model by F1 score** (accuracy as tiebreaker).
6. Persist artifacts under `artifacts/model/`.

### Logged metrics (per model)

| Metric | Description |
| --- | --- |
| `accuracy` | Overall classification accuracy |
| `precision` | Precision for the positive (high-risk) class |
| `recall` | Recall for the positive class |
| `f1` | F1 score for the positive class |
| `roc_auc` | ROC-AUC when probability scores are available |

### Logged parameters (per model)

| Parameter | Description |
| --- | --- |
| `model_type` | `LogisticRegression` or `RandomForestClassifier` |
| `train_size` | Number of training rows |
| `test_size` | Number of test rows |
| `feature_count` | Number of input features (16) |
| `random_state` | Split and model random seed (42) |

## MLflow Tracking

Each training run is logged to the local file store at `./mlruns` under the experiment `vendorrisk-copilot`.

Start the UI from the repo root:

```bash
# PowerShell
$env:MLFLOW_ALLOW_FILE_STORE="true"; mlflow ui

# bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui
```

Or use Docker Compose (`http://127.0.0.1:5000`).

### Logged artifacts (per run)

| Artifact | Purpose |
| --- | --- |
| `feature_names.json` | Ordered list of model input columns |
| `classification_report.txt` | sklearn classification report on the test split |
| `confusion_matrix.json` | Confusion matrix as JSON |
| `model/` | Serialized sklearn model (MLflow sklearn flavor) |

![MLflow experiment runs](../docs/screenshots/mlflow_runs.png)

> Placeholder: capture the MLflow runs list comparing LogisticRegression and RandomForestClassifier with F1 and accuracy metrics.

![MLflow run details](../docs/screenshots/mlflow_run_details.png)

> Placeholder: capture a single run showing parameters, metrics, and the logged artifacts.

## Model Artifacts

After training, these files are written to `artifacts/model/`:

| File | Description |
| --- | --- |
| `vendor_risk_model.joblib` | Champion model bundle (model, feature columns, metrics) |
| `feature_names.json` | Feature list used by `predict_risk` and the API |
| `training_baseline.json` | Mean values for drift-monitored features at training time |
| `model_results.json` | Side-by-side comparison of all trained models |
| `latest_drift_report.json` | Most recent drift check (written by `monitor_drift`) |

`predict_risk` and the FastAPI workflow load `vendor_risk_model.joblib` and `feature_names.json`. If they are missing, training runs automatically on first use.

## Feature Baseline

`training_baseline.json` captures the training-time mean of four operational signals:

- `pending_invoice_amount`
- `sla_breach_rate`
- `compliance_missing_flag`
- `duplicate_invoice_count`

These means represent the expected distribution when the model was trained. Drift monitoring compares fresh `vendor_features.csv` rows against this snapshot.

## Drift Monitoring

Run drift checks after refreshing vendor features:

```bash
python -m src.pipelines.build_features   # optional, if data changed
python -m src.ml.monitor_drift
```

The monitor:

1. Reads `artifacts/model/training_baseline.json`.
2. Reads current `data/processed/vendor_features.csv`.
3. Computes mean per monitored feature and percent change from baseline.
4. Flags drift when absolute percent change exceeds the configured threshold.
5. Writes `artifacts/model/latest_drift_report.json`.

### Drift thresholds

| Feature | Threshold (% change) |
| --- | --- |
| `pending_invoice_amount` | 30% |
| `sla_breach_rate` | 20% |
| `compliance_missing_flag` | 20% |
| `duplicate_invoice_count` | 30% |

### Report format

Each metric row includes:

```json
{
  "metric": "sla_breach_rate",
  "baseline_value": 0.12,
  "current_value": 0.18,
  "percent_change": 50.0,
  "threshold_percent": 20.0,
  "drift_detected": true
}
```

The CLI prints a table summary and saves the full JSON report.

![Drift monitoring output](../docs/screenshots/drift_monitoring.png)

> Placeholder: capture terminal output from `python -m src.ml.monitor_drift` showing the drift table.

## Mapping to Production MLOps

This local setup mirrors patterns used in production ML platforms:

| Local component | Production equivalent |
| --- | --- |
| `python -m src.ml.train_model` | Scheduled training job (Airflow, Azure ML pipeline, SageMaker) |
| `./mlruns` + MLflow UI | MLflow Tracking Server or managed experiment store |
| `vendor_risk_model.joblib` | Model Registry champion artifact or object-store deployment bundle |
| `training_baseline.json` | Feature store statistics or monitoring baseline in Evidently/WhyLabs |
| `monitor_drift` | Scheduled data-quality job with alerting (PagerDuty, Slack) |
| `latest_drift_report.json` | Drift dashboard feed or incident ticket attachment |

A typical production extension path:

1. **CI/CD** — run `pytest` and retrain on labeled data merges; gate promotion on F1 regression.
2. **Model Registry** — register the champion in MLflow Model Registry with stage transitions (`Staging` → `Production`).
3. **Serving** — replace joblib load with a containerized inference endpoint (FastAPI is already in place).
4. **Monitoring** — stream live feature aggregates to a drift service; alert when thresholds breach.
5. **Retraining** — trigger `train_model` when drift persists or new labels arrive.

## Quick Reference

```bash
# Full local MLOps loop
python -m src.pipelines.build_features
python -m src.ml.train_model
$env:MLFLOW_ALLOW_FILE_STORE="true"; mlflow ui
python -m src.ml.monitor_drift
python -m src.ml.predict_risk --vendor-id V001
```
