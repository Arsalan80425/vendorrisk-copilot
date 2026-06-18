"""Generate deterministic synthetic data for the VendorRisk Copilot demo."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.config import CONTRACTS_DIR, RAW_DATA_DIR, VENDOR_FEATURES_PATH, ensure_directories

SEED = 42
REFERENCE_DATE = date(2026, 6, 17)


@dataclass(frozen=True)
class VendorSeed:
    """Input shape used to generate vendor master, invoices, tickets, and contracts."""

    vendor_id: str
    vendor_name: str
    category: str
    country: str
    criticality: str
    compliance_status: str
    account_manager: str
    business_owner: str
    annual_contract_value: int
    contract_start: date
    contract_end: date
    profile: str


def slugify(value: str) -> str:
    """Return a stable file-safe slug for contract file names."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _vendors() -> list[VendorSeed]:
    """Return 25 deterministic vendors with low, medium, and high-risk profiles."""
    return [
        VendorSeed("V001", "DataBridge Solutions", "Data Processing", "USA", "High", "Missing SOC2", "Priya Shah", "Data Engineering", 820000, date(2024, 8, 1), date(2026, 7, 20), "high"),
        VendorSeed("V002", "CloudNova Systems", "Cloud Infrastructure", "USA", "High", "Complete", "Maya Chen", "Infrastructure", 1850000, date(2024, 8, 1), date(2027, 8, 31), "medium"),
        VendorSeed("V003", "Supportly India", "Customer Support", "India", "Low", "Complete", "Arjun Mehta", "Customer Operations", 300000, date(2025, 1, 1), date(2027, 1, 1), "low"),
        VendorSeed("V004", "SecurePay Gateway", "Payments", "USA", "High", "Complete", "Daniel Ruiz", "Finance", 1320000, date(2024, 5, 1), date(2027, 4, 30), "medium"),
        VendorSeed("V005", "MediaTech Studio", "Marketing", "UK", "Low", "Incomplete", "Emma Clarke", "Marketing", 260000, date(2025, 2, 1), date(2026, 7, 10), "high"),
        VendorSeed("V006", "InsightOps Analytics", "Analytics", "USA", "Medium", "Complete", "Noah Wilson", "Revenue Operations", 540000, date(2024, 9, 1), date(2026, 12, 31), "low"),
        VendorSeed("V007", "CyberShield MDR", "Cybersecurity", "USA", "High", "Complete", "Nina Patel", "Security", 960000, date(2025, 1, 1), date(2027, 1, 1), "low"),
        VendorSeed("V008", "CloudScale Backup", "Cloud Infrastructure", "Canada", "Medium", "Expired Insurance", "Leo Martin", "Infrastructure", 610000, date(2024, 3, 15), date(2026, 9, 1), "medium"),
        VendorSeed("V009", "MarketPulse Ads", "Marketing", "USA", "Low", "Complete", "Olivia Green", "Marketing", 340000, date(2025, 4, 1), date(2027, 3, 31), "low"),
        VendorSeed("V010", "LedgerFlow AP", "Payments", "USA", "Medium", "Complete", "Ethan Brooks", "Finance", 470000, date(2024, 11, 1), date(2026, 11, 30), "medium"),
        VendorSeed("V011", "DataCleanse Pro", "Data Processing", "India", "Medium", "Missing DPA", "Ravi Iyer", "Data Engineering", 390000, date(2025, 1, 15), date(2026, 8, 10), "high"),
        VendorSeed("V012", "HelpDesk Hive", "Customer Support", "USA", "Low", "Complete", "Sarah Miller", "Customer Operations", 305000, date(2024, 12, 1), date(2026, 7, 5), "medium"),
        VendorSeed("V013", "PeoplePulse HR", "Analytics", "USA", "Medium", "Complete", "Grace Lee", "People Operations", 425000, date(2025, 6, 1), date(2027, 5, 31), "low"),
        VendorSeed("V014", "AdBeacon Network", "Marketing", "USA", "Medium", "Expired Insurance", "Liam Turner", "Marketing", 510000, date(2024, 7, 1), date(2026, 8, 18), "high"),
        VendorSeed("V015", "FinTrust Audit", "Data Processing", "Ireland", "High", "Complete", "Ava Murphy", "Finance", 690000, date(2025, 3, 1), date(2027, 2, 28), "low"),
        VendorSeed("V016", "InfraWatch Observability", "Cloud Infrastructure", "Germany", "High", "Complete", "Jonas Weber", "Infrastructure", 880000, date(2024, 10, 1), date(2026, 10, 15), "medium"),
        VendorSeed("V017", "BrandLift Creative", "Marketing", "Australia", "Low", "Complete", "Chloe Evans", "Marketing", 215000, date(2025, 7, 1), date(2027, 6, 30), "low"),
        VendorSeed("V018", "PayRoute Connect", "Payments", "Singapore", "High", "Incomplete", "Wei Tan", "Finance", 740000, date(2024, 4, 1), date(2026, 7, 28), "high"),
        VendorSeed("V019", "SafeHarbor GRC", "Cybersecurity", "USA", "Medium", "Complete", "Mason Scott", "Security", 455000, date(2025, 2, 15), date(2027, 2, 14), "low"),
        VendorSeed("V020", "TicketNest CX", "Customer Support", "Mexico", "Medium", "Missing SOC2", "Sofia Garcia", "Customer Operations", 380000, date(2024, 6, 1), date(2026, 8, 1), "high"),
        VendorSeed("V021", "QueryForge BI", "Analytics", "USA", "Medium", "Complete", "Henry Adams", "Revenue Operations", 575000, date(2025, 5, 1), date(2027, 4, 30), "low"),
        VendorSeed("V022", "ObjectVault Storage", "Cloud Infrastructure", "USA", "High", "Complete", "Lily Foster", "Infrastructure", 990000, date(2024, 9, 15), date(2026, 12, 15), "medium"),
        VendorSeed("V023", "TrustDesk KYC", "Data Processing", "Netherlands", "High", "Missing DPA", "Mila Janssen", "Compliance", 675000, date(2024, 1, 10), date(2026, 7, 25), "high"),
        VendorSeed("V024", "ReviewRocket CX", "Customer Support", "USA", "Low", "Complete", "Jack Young", "Customer Operations", 245000, date(2025, 8, 1), date(2027, 7, 31), "low"),
        VendorSeed("V025", "CampaignWorks AI", "Marketing", "USA", "Medium", "Complete", "Ella Brown", "Marketing", 430000, date(2025, 9, 1), date(2027, 8, 31), "medium"),
    ]


def _contract_text(vendor: VendorSeed) -> str:
    """Build a realistic contract text document for one vendor."""
    payment_days = 30 if vendor.profile != "high" else 45
    sla_target = "99.9%" if vendor.category in {"Cloud Infrastructure", "Payments", "Cybersecurity"} else "95%"
    renewal_notice = 60 if vendor.criticality == "High" else 30
    return f"""{vendor.vendor_name} Master Services Agreement

Payment Terms
Invoices are billed monthly in USD-equivalent currency with net {payment_days} payment terms. Undisputed invoices must be paid by the due date, and disputed invoices require written notice with supporting purchase-order detail.

SLA Obligations
Vendor will maintain {sla_target} service availability where applicable and acknowledge critical incidents within the contracted SLA window. Repeated SLA misses require a remediation plan and executive review.

Compliance Requirements
Vendor must maintain current compliance evidence appropriate to the service, including SOC2, insurance, privacy addenda, and data processing agreements when required. Missing or expired evidence must be remediated before renewal approval.

Renewal Terms
The contract end date is {vendor.contract_end.isoformat()}. Renewal requires business-owner approval and non-renewal notice at least {renewal_notice} days before expiration unless a signed order form states otherwise.

Termination Rights
Customer may terminate for uncured material breach, repeated service failures, compliance failure, insolvency, or unauthorized data use after the applicable cure period.

Data Security
Vendor must encrypt customer data in transit and at rest, enforce least-privilege access, retain audit logs, and notify the customer of material security incidents within 48 hours.

Escalation Procedure
Operational escalations start with the account manager, then move to executive sponsorship if a Severity Critical issue remains unresolved beyond the SLA window. Vendor must provide written status updates until closure.
"""


def _invoice_rows(vendors: list[VendorSeed], rng: random.Random) -> list[dict[str, object]]:
    """Generate 120 invoice rows with intentional risk patterns."""
    rows: list[dict[str, object]] = []
    invoice_counter = 1001
    currencies = {"USA": "USD", "India": "USD", "UK": "GBP", "Canada": "CAD", "Ireland": "EUR", "Germany": "EUR", "Australia": "AUD", "Singapore": "USD", "Mexico": "USD", "Netherlands": "EUR"}
    cost_centers = ["IT-OPS", "FIN-AP", "DATA-ENG", "CX-OPS", "MKT-GROWTH", "SEC-GRC", "REV-OPS"]

    for index, vendor in enumerate(vendors):
        invoice_count = 5 if index < 20 else 4
        monthly_amount = max(7500, round(vendor.annual_contract_value / 12 / 100) * 100)
        for offset in range(invoice_count):
            invoice_date = date(2026, 1, 5) + timedelta(days=offset * 31 + rng.randint(0, 6))
            amount = int(monthly_amount * rng.uniform(0.85, 1.18))
            status = "Paid"
            po_match = "Yes"
            paid_date = invoice_date + timedelta(days=rng.randint(10, 29))

            if vendor.profile == "high" and offset >= invoice_count - 2:
                status = "Pending" if offset == invoice_count - 2 else "Overdue"
                po_match = "No" if offset % 2 == 0 else "Yes"
                amount = int(monthly_amount * rng.uniform(1.5, 2.3))
                paid_date = None
            elif vendor.profile == "medium" and offset == invoice_count - 1:
                status = "Pending"
                amount = int(monthly_amount * rng.uniform(1.0, 1.35))
                po_match = "No" if vendor.vendor_id in {"V008", "V010", "V025"} else "Yes"
                paid_date = None

            rows.append(
                {
                    "invoice_id": f"INV-{invoice_counter}",
                    "vendor_id": vendor.vendor_id,
                    "invoice_date": invoice_date.isoformat(),
                    "amount": amount,
                    "due_date": (invoice_date + timedelta(days=30)).isoformat(),
                    "paid_date": paid_date.isoformat() if paid_date else "",
                    "invoice_status": status,
                    "po_match": po_match,
                    "currency": currencies.get(vendor.country, "USD"),
                    "cost_center": rng.choice(cost_centers),
                }
            )
            invoice_counter += 1

    # Force the flagship high-risk pattern: duplicate pending invoices and high pending exposure.
    for row in rows:
        if row["vendor_id"] == "V001" and row["invoice_id"] in {"INV-1004", "INV-1005"}:
            row["invoice_date"] = "2026-04-10"
            row["amount"] = 185000
            row["due_date"] = "2026-05-10"
            row["paid_date"] = ""
            row["invoice_status"] = "Pending"
            row["po_match"] = "No"
            row["cost_center"] = "DATA-ENG"
    return rows


def _ticket_rows(vendors: list[VendorSeed], rng: random.Random) -> list[dict[str, object]]:
    """Generate 90 support-ticket rows with clean, medium, and severe SLA patterns."""
    rows: list[dict[str, object]] = []
    ticket_counter = 2001
    issue_types = ["Availability", "Data Delay", "Access", "Billing", "Security Review", "Integration", "Reporting"]

    for index, vendor in enumerate(vendors):
        ticket_count = 4 if index < 15 else 3
        for offset in range(ticket_count):
            created_date = date(2026, 1, 10) + timedelta(days=offset * 37 + rng.randint(0, 8))
            severity = rng.choice(["Low", "Medium", "High"])
            sla_hours = {"Low": 24, "Medium": 12, "High": 6, "Critical": 2}[severity]
            resolution_hours = rng.randint(1, max(sla_hours - 1, 1))
            status = "Resolved"

            if vendor.profile == "high" and offset >= ticket_count - 2:
                severity = "Critical" if offset == ticket_count - 1 else "High"
                sla_hours = 2 if severity == "Critical" else 6
                resolution_hours = sla_hours + rng.randint(8, 24)
                status = "Escalated" if severity == "Critical" else "Resolved"
            elif vendor.profile == "medium" and offset == ticket_count - 1:
                severity = "High"
                sla_hours = 6
                resolution_hours = sla_hours + rng.randint(2, 8)
                status = "Resolved"

            if vendor.vendor_id == "V002" and offset == ticket_count - 1:
                severity = "High"
                sla_hours = 6
                resolution_hours = 11
                status = "Resolved"
            if vendor.vendor_id == "V003":
                severity = rng.choice(["Low", "Medium"])
                sla_hours = 24 if severity == "Low" else 12
                resolution_hours = rng.randint(1, sla_hours - 1)
                status = "Resolved"

            rows.append(
                {
                    "ticket_id": f"TCK-{ticket_counter}",
                    "vendor_id": vendor.vendor_id,
                    "created_date": created_date.isoformat(),
                    "severity": severity,
                    "resolution_hours": resolution_hours,
                    "sla_hours": sla_hours,
                    "status": status,
                    "issue_type": rng.choice(issue_types),
                }
            )
            ticket_counter += 1
    return rows


def generate_synthetic_data(seed: int = SEED) -> dict[str, int]:
    """Generate all synthetic CSV and contract artifacts."""
    rng = random.Random(seed)
    ensure_directories()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)

    vendors = _vendors()
    vendor_rows = [
        {
            "vendor_id": vendor.vendor_id,
            "vendor_name": vendor.vendor_name,
            "category": vendor.category,
            "country": vendor.country,
            "criticality": vendor.criticality,
            "contract_start": vendor.contract_start.isoformat(),
            "contract_end": vendor.contract_end.isoformat(),
            "compliance_status": vendor.compliance_status,
            "account_manager": vendor.account_manager,
            "business_owner": vendor.business_owner,
            "annual_contract_value": vendor.annual_contract_value,
        }
        for vendor in vendors
    ]
    invoices = _invoice_rows(vendors, rng)
    tickets = _ticket_rows(vendors, rng)

    pd.DataFrame(vendor_rows).to_csv(RAW_DATA_DIR / "vendor_master.csv", index=False)
    pd.DataFrame(invoices).to_csv(RAW_DATA_DIR / "invoices.csv", index=False)
    pd.DataFrame(tickets).to_csv(RAW_DATA_DIR / "support_tickets.csv", index=False)

    for existing_contract in CONTRACTS_DIR.glob("*.txt"):
        existing_contract.unlink()
    for vendor in vendors:
        contract_path = CONTRACTS_DIR / f"{slugify(vendor.vendor_name)}_contract.txt"
        contract_path.write_text(_contract_text(vendor), encoding="utf-8")

    if VENDOR_FEATURES_PATH.exists():
        VENDOR_FEATURES_PATH.unlink()

    return {
        "vendors": len(vendor_rows),
        "invoices": len(invoices),
        "support_tickets": len(tickets),
        "contracts": len(vendors),
    }


def main() -> None:
    """CLI entrypoint for synthetic data generation."""
    counts = generate_synthetic_data()
    print("Generated deterministic synthetic data:")
    for name, count in counts.items():
        print(f"- {name}: {count}")
    print(f"Raw CSVs saved to {RAW_DATA_DIR}")
    print(f"Contract files saved to {CONTRACTS_DIR}")


if __name__ == "__main__":
    main()
