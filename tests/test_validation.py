import pandas as pd

from src.data_quality.validate_data import validate_all, validate_invoices, validate_vendor_master


def test_validation_passes_with_expected_warnings():
    report = validate_all()

    assert report.passed is True
    assert report.errors == []
    messages = {issue.message for issue in report.warnings}
    assert "Vendors have missing or incomplete compliance evidence" in messages
    assert "Potential duplicate invoices found" in messages
    assert report.summary["duplicate_invoice_rows"] == 2


def test_duplicate_invoice_detection():
    invoices = pd.DataFrame(
        [
            {
                "invoice_id": "INV-1",
                "vendor_id": "V001",
                "invoice_date": "2026-01-01",
                "due_date": "2026-01-31",
                "paid_date": "",
                "amount": 1000,
                "invoice_status": "Pending",
                "po_match": "Yes",
                "currency": "USD",
                "cost_center": "DATA-ENG",
            },
            {
                "invoice_id": "INV-2",
                "vendor_id": "V001",
                "invoice_date": "2026-01-01",
                "due_date": "2026-01-31",
                "paid_date": "",
                "amount": 1000,
                "invoice_status": "Pending",
                "po_match": "No",
                "currency": "USD",
                "cost_center": "DATA-ENG",
            },
        ]
    )

    errors, warnings = validate_invoices(invoices, {"V001"})

    assert errors == []
    assert any(issue.message == "Potential duplicate invoices found" for issue in warnings)


def test_invalid_contract_date_validation():
    vendors = pd.DataFrame(
        [
            {
                "vendor_id": "V001",
                "vendor_name": "Bad Dates Inc",
                "category": "Analytics",
                "criticality": "Medium",
                "compliance_status": "Complete",
                "contract_start": "2026-12-31",
                "contract_end": "2026-01-01",
                "account_manager": "Alex Rivera",
                "business_owner": "Procurement",
                "annual_contract_value": 100000,
            }
        ]
    )

    errors, _ = validate_vendor_master(vendors)

    assert any(issue.field == "contract_end" for issue in errors)
