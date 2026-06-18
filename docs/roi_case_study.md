# ROI Case Study — Northwind Analytics Procurement Pilot

## Company Context

**Northwind Analytics** is a mid-market B2B SaaS company with **12 active vendors** across cloud infrastructure, data processing, customer support, payments, and marketing. Finance processes roughly **40 invoices per quarter** through a shared AP queue while procurement tracks renewals and compliance in spreadsheets.

Before VendorRisk Copilot, weekly vendor reviews took 6–8 analyst hours and still missed cross-system signals: AP saw duplicate invoices, support saw SLA breaches, and legal held contracts — but no one connected them at the vendor level.

---

## Portfolio Snapshot

| Metric | Value |
| --- | --- |
| Active vendors | 12 |
| Invoices (current quarter) | 40 |
| Total annual vendor spend | $4,860,000 |
| Vendors with compliance gaps | 3 |
| Vendors renewing within 90 days | 4 |
| SLA breach events (90 days) | 9 |
| Duplicate invoice candidates | 5 |

### Vendor Roster (abbreviated)

| Vendor | Category | Annual Value | Key Risk Signal |
| --- | --- | --- | --- |
| DataBridge Solutions | Data Processing | $820,000 | Missing SOC2, duplicate pending invoices, renewal in 33 days |
| CloudNova Systems | Cloud Infrastructure | $1,850,000 | SLA breaches on critical incidents |
| Supportly India | Customer Support | $300,000 | Clean profile — control vendor |
| MediaTech Studio | Marketing | $260,000 | Incomplete compliance, renewal in 23 days |
| SecurePay Gateway | Payments | $1,320,000 | High criticality, one PO mismatch |
| HelpDesk Hive | Customer Support | $305,000 | Near-term renewal |
| DataCleanse Pro | Data Processing | $390,000 | Missing DPA |
| CloudScale Backup | Cloud Infrastructure | $610,000 | Expired insurance certificate |
| LedgerFlow AP | Payments | $470,000 | Pending invoice backlog |
| InsightOps Analytics | Analytics | $540,000 | Low risk |
| AdBeacon Network | Marketing | $510,000 | Expired insurance, SLA breaches |
| PayRoute Connect | Payments | $740,000 | Incomplete compliance, renewal in 41 days |

---

## Exposure Analysis

VendorRisk Copilot applies the same exposure formulas used in the Streamlit ROI simulator.

### 1. Duplicate Invoice Exposure — $48,750

Three vendors show duplicate invoice candidates against pending amounts:

| Vendor | Pending AP | Duplicate Count | Estimated Duplicate Exposure |
| --- | --- | --- | --- |
| DataBridge Solutions | $370,000 | 2 | $24,667 |
| LedgerFlow AP | $118,000 | 2 | $19,667 |
| AdBeacon Network | $28,500 | 1 | $4,416 |

**Formula:** `pending_invoice_amount × min(duplicate_count, 3) / invoice_count`

Duplicate exposure is often discovered only at month-end close. Early detection prevents double payment on overlapping invoice numbers and amounts.

### 2. SLA Penalty Exposure — $112,400

Nine SLA breach events across four vendors trigger contractual penalty estimates at 1% of annual contract value per breach (minimum $5,000):

| Vendor | Breaches | Annual Value | Penalty Estimate |
| --- | --- | --- | --- |
| CloudNova Systems | 3 | $1,850,000 | $55,500 |
| DataBridge Solutions | 2 | $820,000 | $16,400 |
| AdBeacon Network | 2 | $510,000 | $10,200 |
| CloudScale Backup | 2 | $610,000 | $12,200 |
| PayRoute Connect | 0 | — | — |

**Formula:** `sla_breach_count × max(annual_contract_value × 1%, $5,000)`

Support dashboards showed incidents, but procurement had no aggregated penalty view tied to contract SLA clauses.

### 3. Renewal Exposure — $136,800

Four vendors have contracts expiring within 90 days. Renewal risk is estimated at 4% of annual value when compliance or performance issues exist:

| Vendor | Days to Renewal | Annual Value | Renewal Risk Estimate |
| --- | --- | --- | --- |
| MediaTech Studio | 23 | $260,000 | $10,400 |
| HelpDesk Hive | 18 | $305,000 | $12,200 |
| DataBridge Solutions | 33 | $820,000 | $32,800 |
| PayRoute Connect | 41 | $740,000 | $29,600 |
| CloudScale Backup | 76 | $610,000 | $24,400 |
| AdBeacon Network | 62 | $510,000 | $20,400 |

**Formula:** `renewal_within_90_days × annual_contract_value × 4%`

*(Four vendors flagged; two additional near-renewal vendors included in the 90-day window for this pilot narrative.)*

### 4. Pending and Overdue Invoice Exposure — $612,000

Open AP across medium- and high-risk vendors represents cash at risk if approvals proceed without review:

| Status | Amount |
| --- | --- |
| Pending invoices | $498,000 |
| Overdue invoices | $114,000 |
| **Subtotal** | **$612,000** |

---

## Total Estimated Preventable Exposure

| Exposure Bucket | Amount |
| --- | --- |
| Duplicate invoice exposure | $48,750 |
| SLA penalty exposure | $112,400 |
| Renewal exposure | $136,800 |
| Pending / overdue exposure | $612,000 |
| **Total estimated preventable exposure** | **$909,950** |

> **Executive summary:** Northwind can address nearly **$910K** in identifiable procurement risk across duplicate payments, SLA penalties, renewal decisions, and open invoice backlog — before considering analyst time savings.

---

## How the Procurement Team Uses the System

### Monday — Portfolio standup (15 minutes)

1. Open the **Streamlit dashboard** executive summary.
2. Review KPIs: high-risk vendor count, pending exposure, contracts expiring in 90 days.
3. Walk the **high-risk review queue** and assign owners for the week.

### Tuesday — Vendor deep dive on DataBridge Solutions

1. Select **V001 — DataBridge Solutions** in the vendor explorer.
2. Click **Analyze Vendor** to run the LangGraph workflow.
3. Review **top risk factors**: duplicate invoice candidates, SLA breaches, missing SOC2, renewal within 33 days.
4. Inspect **contract evidence** retrieved from the DataBridge MSA (payment terms, SLA obligations, compliance requirements, renewal notice).
5. Read the **recommendation**: escalate to procurement manager and block approval until review.
6. Export the **automation payload** to n8n.

### Wednesday — Automation and stakeholder notification

1. n8n workflow receives the payload with `event_type: vendor_risk_review_required`.
2. Slack message goes to the business owner (Data Engineering) and account manager with exposure and factors.
3. A Jira or ServiceNow task is opened for compliance remediation (SOC2 evidence).

### Thursday — Renewal committee prep

1. Filter the **renewal timeline** chart for contracts expiring within 90 days.
2. Cross-reference compliance status distribution.
3. Block MediaTech and PayRoute renewals until compliance artifacts are uploaded.

### Friday — Model and drift review

1. Open **MLflow UI** to confirm the latest RandomForest run and F1 score.
2. Run `python -m src.ml.monitor_drift` after new invoice data lands.
3. Document changes in the weekly procurement risk memo.

---

## Outcomes After 90 Days (Projected)

| Outcome | Estimate |
| --- | --- |
| Duplicate payments avoided | $48,750 |
| SLA credits / penalties recovered or avoided | $45,000 |
| Renewal renegotiations with compliance gates | $55,000 value protected |
| Analyst hours saved (6 hrs/week × $85/hr × 13 weeks) | $6,630 |
| **90-day gross benefit** | **~$155,000** |
| Platform automation cost (amortized) | $16,250 |
| **90-day net benefit** | **~$139,000** |

---

## Why This Resonates With Leadership

VendorRisk Copilot reframes procurement from reactive spreadsheet review to a **continuous operating signal**:

- One vendor record connects invoices, tickets, contracts, and ML score
- Contract RAG grounds recommendations in actual clause text
- Exposure numbers translate technical findings into finance language
- n8n payloads close the loop from insight to action

The Northwind pilot uses a simplified **12-vendor / 40-invoice** scope for storytelling. The repository ships a larger deterministic corpus (25 vendors, 120 invoices) for richer dashboard demos — the same formulas and workflow apply at either scale.
