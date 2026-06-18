"""Simulate n8n workflow: Render API -> IF High -> Slack."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

RENDER_URL = "https://vendorrisk-copilot.onrender.com/analyze-vendor"


def main() -> None:
    # Step 1: Render API
    req = urllib.request.Request(
        RENDER_URL,
        data=json.dumps({"vendor_id": "V001"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    print(f"1. Render API: {data['vendor_id']} | {data['risk_level']} | score={data['risk_score']}")

    # Step 2: IF High Risk
    is_high = data["risk_level"] == "High"
    print(f"2. IF High Risk: {'TRUE -> Slack + Sheets' if is_high else 'FALSE branch'}")
    if not is_high:
        raise SystemExit("Expected V001 to route through High risk branch")

    # Step 3: Slack Alert
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook:
        print("3. Slack: SKIPPED (set SLACK_WEBHOOK_URL in n8n env or .env)")
    else:
        slack_body = json.dumps(
            {
                "text": (
                    f"High-Risk Vendor Alert\n\n"
                    f"Vendor: {data['vendor_name']}\n"
                    f"Risk Level: {data['risk_level']}\n"
                    f"Risk Score: {data['risk_score']}\n"
                    f"Estimated Exposure: ${data['estimated_financial_exposure']}\n\n"
                    f"Recommended Action:\n{data['recommended_action']}"
                )
            }
        ).encode()
        slack_req = urllib.request.Request(
            webhook,
            data=slack_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(slack_req, timeout=30) as resp:
                body = resp.read().decode()
                print(f"3. Slack: HTTP {resp.status} | {body}")
        except urllib.error.HTTPError as exc:
            print(f"3. Slack: HTTP {exc.code} | {exc.read().decode()}")

    print("4. Google Sheets: configure YOUR_GOOGLE_SHEET_ID + Google OAuth in n8n UI, then re-import workflow")


if __name__ == "__main__":
    main()
