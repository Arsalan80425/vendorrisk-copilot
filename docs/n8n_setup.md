# n8n Setup

## Import the workflow

1. Open n8n.
2. Go to **Workflows**.
3. Choose **Import from File**.
4. Select `n8n/vendor_risk_workflow.example.json`.
5. Save the workflow as your own copy before editing.

The example uses a Manual Trigger and sends sample `vendor_id = V001` to the VendorRisk Copilot FastAPI endpoint.

## Use the deployed Render API

For n8n Cloud or any external n8n instance, point the **Analyze Vendor via FastAPI** HTTP Request node at the public Render service:

```text
https://vendorrisk-copilot.onrender.com/analyze-vendor
```

The checked-in workflow JSON already uses this URL. Set `DEPLOYMENT_MODE=lightweight` on Render so the free tier stays within memory limits.

Verify the API before running the workflow:

```bash
curl https://vendorrisk-copilot.onrender.com/health
curl -X POST https://vendorrisk-copilot.onrender.com/analyze-vendor \
  -H "Content-Type: application/json" \
  -d "{\"vendor_id\":\"V001\"}"
```

`V001` (DataBridge Solutions) returns `risk_level: High`, which exercises the IF branch, Slack alert, and Google Sheets append.

## Run FastAPI locally

From the project root:

```bash
python -m src.data_generation.generate_synthetic_data
python -m src.pipelines.build_features
python -m src.rag.ingest_contracts
python -m src.ml.train_model
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

If n8n runs in Docker on the same machine, use:

```text
http://host.docker.internal:8000/analyze-vendor
```

If n8n runs directly on your host machine, use:

```text
http://127.0.0.1:8000/analyze-vendor
```

## Use n8n Cloud with ngrok

n8n Cloud cannot call your private localhost URL directly. Expose FastAPI with ngrok:

```bash
ngrok http 8000
```

Copy the HTTPS forwarding URL and update the HTTP Request node:

```text
https://your-ngrok-subdomain.ngrok-free.app/analyze-vendor
```

Keep FastAPI and ngrok running while testing the cloud workflow.

## Connect Slack

The workflow includes a disabled Slack webhook placeholder. To use it:

1. Create an incoming webhook in Slack.
2. Store the webhook URL as an n8n environment variable named `SLACK_WEBHOOK_URL`.
3. Enable the **Send High-Risk Slack Alert** HTTP Request node.
4. Run the workflow with a high-risk vendor such as `V001`.

Do not paste real Slack webhook URLs into exported example JSON files.

## Connect Google Sheets

To log vendor risk results:

1. Add a Google Sheets node after either branch.
2. Connect your Google account in n8n credentials.
3. Map fields such as `vendor_name`, `risk_score`, `risk_level`, `recommended_action`, and `estimated_financial_exposure`.
4. Append rows to a sheet dedicated to procurement risk review.

For a public portfolio export, remove credential IDs before sharing.

## Export sanitized workflow JSON

Before committing or sharing an n8n workflow:

1. Remove real webhook URLs, API keys, credential IDs, and OAuth references.
2. Replace secrets with environment-variable references such as `{{$env.SLACK_WEBHOOK_URL}}`.
3. Disable placeholder nodes that would call external systems.
4. Export the workflow JSON from n8n.
5. Save it as `n8n/vendor_risk_workflow.example.json`.

The checked-in example is intentionally sanitized and should not contain live secrets.
