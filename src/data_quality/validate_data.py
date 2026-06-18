"""Data quality checks for raw procurement and vendor-risk datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR
from src.utils.schemas import DataQualityIssue, DataQualityReport

ALLOWED_CRITICALITY = {"Low", "Medium", "High"}
ALLOWED_COMPLIANCE = {"Complete", "Missing SOC2", "Expired Insurance", "Missing DPA", "Incomplete"}
ALLOWED_INVOICE_STATUS = {"Paid", "Pending", "Overdue"}
ALLOWED_PO_MATCH = {"Yes", "No"}
ALLOWED_SEVERITY = {"Low", "Medium", "High", "Critical"}
ALLOWED_TICKET_STATUS = {"Resolved", "Open", "Escalated"}

REQUIRED_COLUMNS = {
    "vendor_master.csv": {
        "vendor_id",
        "vendor_name",
        "category",
        "criticality",
        "compliance_status",
        "contract_start",
        "contract_end",
        "account_manager",
        "business_owner",
        "annual_contract_value",
    },
    "invoices.csv": {
        "invoice_id",
        "vendor_id",
        "invoice_date",
        "due_date",
        "paid_date",
        "amount",
        "invoice_status",
        "po_match",
        "currency",
        "cost_center",
    },
    "support_tickets.csv": {
        "ticket_id",
        "vendor_id",
        "created_date",
        "severity",
        "resolution_hours",
        "sla_hours",
        "status",
        "issue_type",
    },
}


def _read_csv(raw_data_dir: Path, name: str) -> pd.DataFrame:
    path = raw_data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required raw dataset: {path}")
    return pd.read_csv(path)


def _missing_required_columns(df: pd.DataFrame, dataset: str) -> list[DataQualityIssue]:
    missing = REQUIRED_COLUMNS[dataset].difference(df.columns)
    if not missing:
        return []
    return [
        DataQualityIssue(
            dataset=dataset,
            severity="error",
            message=f"Missing columns: {sorted(missing)}",
            affected_rows=len(df),
        )
    ]


def _not_null_issue(df: pd.DataFrame, dataset: str, column: str) -> DataQualityIssue | None:
    missing = df[column].isna() | df[column].astype(str).str.strip().eq("")
    if not missing.any():
        return None
    return DataQualityIssue(
        dataset=dataset,
        severity="error",
        field=column,
        message=f"{column} must not be null",
        affected_rows=int(missing.sum()),
    )


def _allowed_values_issue(
    df: pd.DataFrame, dataset: str, column: str, allowed_values: set[str]
) -> DataQualityIssue | None:
    invalid = ~df[column].isin(allowed_values)
    if not invalid.any():
        return None
    return DataQualityIssue(
        dataset=dataset,
        severity="error",
        field=column,
        message=f"{column} must be one of {sorted(allowed_values)}",
        affected_rows=int(invalid.sum()),
    )


def validate_vendor_master(vendors: pd.DataFrame) -> tuple[list[DataQualityIssue], list[DataQualityIssue]]:
    """Validate vendor master records."""
    dataset = "vendor_master.csv"
    errors: list[DataQualityIssue] = _missing_required_columns(vendors, dataset)
    warnings: list[DataQualityIssue] = []
    if errors:
        return errors, warnings

    for column in ("vendor_id", "vendor_name"):
        issue = _not_null_issue(vendors, dataset, column)
        if issue:
            errors.append(issue)

    for column, allowed in (
        ("criticality", ALLOWED_CRITICALITY),
        ("compliance_status", ALLOWED_COMPLIANCE),
    ):
        issue = _allowed_values_issue(vendors, dataset, column, allowed)
        if issue:
            errors.append(issue)

    starts = pd.to_datetime(vendors["contract_start"], errors="coerce")
    ends = pd.to_datetime(vendors["contract_end"], errors="coerce")
    invalid_dates = starts.isna() | ends.isna() | ends.le(starts)
    if invalid_dates.any():
        errors.append(
            DataQualityIssue(
                dataset=dataset,
                severity="error",
                field="contract_end",
                message="contract_end must be after contract_start",
                affected_rows=int(invalid_dates.sum()),
            )
        )

    non_complete = vendors["compliance_status"].ne("Complete")
    if non_complete.any():
        warnings.append(
            DataQualityIssue(
                dataset=dataset,
                severity="warning",
                field="compliance_status",
                message="Vendors have missing or incomplete compliance evidence",
                affected_rows=int(non_complete.sum()),
            )
        )
    return errors, warnings


def validate_invoices(
    invoices: pd.DataFrame, known_vendor_ids: set[str] | None = None
) -> tuple[list[DataQualityIssue], list[DataQualityIssue]]:
    """Validate invoice records and identify duplicate invoice candidates."""
    dataset = "invoices.csv"
    errors: list[DataQualityIssue] = _missing_required_columns(invoices, dataset)
    warnings: list[DataQualityIssue] = []
    if errors:
        return errors, warnings

    for column in ("invoice_id", "vendor_id"):
        issue = _not_null_issue(invoices, dataset, column)
        if issue:
            errors.append(issue)

    amounts = pd.to_numeric(invoices["amount"], errors="coerce")
    invalid_amount = amounts.isna() | amounts.le(0)
    if invalid_amount.any():
        errors.append(
            DataQualityIssue(
                dataset=dataset,
                severity="error",
                field="amount",
                message="amount must be greater than 0",
                affected_rows=int(invalid_amount.sum()),
            )
        )

    for column, allowed in (
        ("invoice_status", ALLOWED_INVOICE_STATUS),
        ("po_match", ALLOWED_PO_MATCH),
    ):
        issue = _allowed_values_issue(invoices, dataset, column, allowed)
        if issue:
            errors.append(issue)

    invoice_dates = pd.to_datetime(invoices["invoice_date"], errors="coerce")
    paid_dates = pd.to_datetime(invoices["paid_date"], errors="coerce")
    paid_date_exists = invoices["paid_date"].notna() & invoices["paid_date"].astype(str).str.strip().ne("")
    invalid_paid_dates = paid_date_exists & (paid_dates.isna() | invoice_dates.isna() | paid_dates.lt(invoice_dates))
    if invalid_paid_dates.any():
        errors.append(
            DataQualityIssue(
                dataset=dataset,
                severity="error",
                field="paid_date",
                message="paid_date cannot be before invoice_date when paid_date exists",
                affected_rows=int(invalid_paid_dates.sum()),
            )
        )

    if known_vendor_ids is not None:
        unknown_vendors = ~invoices["vendor_id"].isin(known_vendor_ids)
        if unknown_vendors.any():
            errors.append(
                DataQualityIssue(
                    dataset=dataset,
                    severity="error",
                    field="vendor_id",
                    message="Invoices reference unknown vendors",
                    affected_rows=int(unknown_vendors.sum()),
                )
            )

    duplicate_mask = invoices.duplicated(subset=["vendor_id", "invoice_date", "amount"], keep=False)
    if duplicate_mask.any():
        warnings.append(
            DataQualityIssue(
                dataset=dataset,
                severity="warning",
                message="Potential duplicate invoices found",
                affected_rows=int(duplicate_mask.sum()),
            )
        )
    return errors, warnings


def validate_support_tickets(
    tickets: pd.DataFrame, known_vendor_ids: set[str] | None = None
) -> tuple[list[DataQualityIssue], list[DataQualityIssue]]:
    """Validate support ticket records."""
    dataset = "support_tickets.csv"
    errors: list[DataQualityIssue] = _missing_required_columns(tickets, dataset)
    warnings: list[DataQualityIssue] = []
    if errors:
        return errors, warnings

    for column in ("ticket_id", "vendor_id"):
        issue = _not_null_issue(tickets, dataset, column)
        if issue:
            errors.append(issue)

    for column, allowed in (
        ("severity", ALLOWED_SEVERITY),
        ("status", ALLOWED_TICKET_STATUS),
    ):
        issue = _allowed_values_issue(tickets, dataset, column, allowed)
        if issue:
            errors.append(issue)

    resolution_hours = pd.to_numeric(tickets["resolution_hours"], errors="coerce")
    invalid_resolution = resolution_hours.isna() | resolution_hours.lt(0)
    if invalid_resolution.any():
        errors.append(
            DataQualityIssue(
                dataset=dataset,
                severity="error",
                field="resolution_hours",
                message="resolution_hours must be greater than or equal to 0",
                affected_rows=int(invalid_resolution.sum()),
            )
        )

    sla_hours = pd.to_numeric(tickets["sla_hours"], errors="coerce")
    invalid_sla = sla_hours.isna() | sla_hours.le(0)
    if invalid_sla.any():
        errors.append(
            DataQualityIssue(
                dataset=dataset,
                severity="error",
                field="sla_hours",
                message="sla_hours must be greater than 0",
                affected_rows=int(invalid_sla.sum()),
            )
        )

    if known_vendor_ids is not None:
        unknown_vendors = ~tickets["vendor_id"].isin(known_vendor_ids)
        if unknown_vendors.any():
            errors.append(
                DataQualityIssue(
                    dataset=dataset,
                    severity="error",
                    field="vendor_id",
                    message="Tickets reference unknown vendors",
                    affected_rows=int(unknown_vendors.sum()),
                )
            )
    return errors, warnings


def validate_all(raw_data_dir: Path = RAW_DATA_DIR) -> DataQualityReport:
    """Run schema, row-level, and business-risk data checks for all raw CSVs."""
    frames: dict[str, pd.DataFrame] = {
        name: _read_csv(raw_data_dir, name) for name in REQUIRED_COLUMNS
    }
    vendors = frames["vendor_master.csv"]
    invoices = frames["invoices.csv"]
    tickets = frames["support_tickets.csv"]
    known_vendor_ids = set(vendors["vendor_id"].dropna().astype(str))

    errors: list[DataQualityIssue] = []
    warnings: list[DataQualityIssue] = []
    for dataset, df in frames.items():
        if df.empty:
            errors.append(
                DataQualityIssue(dataset=dataset, severity="error", message="Dataset is empty")
            )

    vendor_errors, vendor_warnings = validate_vendor_master(vendors)
    invoice_errors, invoice_warnings = validate_invoices(invoices, known_vendor_ids)
    ticket_errors, ticket_warnings = validate_support_tickets(tickets, known_vendor_ids)

    errors.extend([*vendor_errors, *invoice_errors, *ticket_errors])
    warnings.extend([*vendor_warnings, *invoice_warnings, *ticket_warnings])

    duplicate_count = 0
    if REQUIRED_COLUMNS["invoices.csv"].issubset(invoices.columns):
        duplicate_count = int(
            invoices.duplicated(subset=["vendor_id", "invoice_date", "amount"], keep=False).sum()
        )

    summary = {
        "datasets": {
            name: {"rows": int(len(df)), "columns": int(len(df.columns))}
            for name, df in frames.items()
        },
        "error_count": len(errors),
        "warning_count": len(warnings),
        "duplicate_invoice_rows": duplicate_count,
    }
    return DataQualityReport(passed=not errors, errors=errors, warnings=warnings, summary=summary)


def main() -> None:
    """CLI entrypoint for data quality validation."""
    report = validate_all()
    print(report.model_dump_json(indent=2))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
