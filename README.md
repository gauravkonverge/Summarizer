# Bedrock Conversation Summarizer

A FastAPI service that removes sensitive information from conversation messages
with Amazon Bedrock Guardrails, generates a summary with an approved Bedrock
model, and verifies the result in a second model pass.

The application supports:

- `brief`, `detailed`, and `bullet_points` summaries
- PII masking before any message is sent to the foundation model
- OAuth2/OIDC bearer-token authentication
- confidence scores, timeline metrics, and configurable cost estimates
- CloudWatch logs, metrics, and alarms for EC2 deployments

## Repository layout

```text
.
├── docs/
│   └── deployment.md # Detailed production deployment guide
├── main/
│   ├── app/          # FastAPI application and summarization pipeline
│   ├── deploy/       # EC2, Nginx, IAM, and CloudWatch templates
│   ├── devtools/     # Local development utilities
│   ├── tests/        # Automated tests
│   ├── .env.example  # Configuration template
│   └── requirements.txt
└── README.md
```

Application commands must be run from the `main` directory unless stated
otherwise.

## Prerequisites

- Python 3.10 or newer
- An AWS account with access to an approved Amazon Bedrock model
- An Amazon Bedrock Guardrail configured to mask sensitive information
- AWS credentials for local live-inference testing, preferably through AWS SSO
  or another temporary-credential workflow

Tests use fake providers and do not require AWS credentials.

## Local setup

From the repository root:

```bash
cd main
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, activate and use the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Configuration

Open `main/.env` and set at least:

```dotenv
APP_ENV=local
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=your-approved-model-or-inference-profile-id
BEDROCK_GUARDRAIL_ID=your-guardrail-id
BEDROCK_GUARDRAIL_VERSION=1
AUTH_ENABLED=false
```

For local AWS authentication, use the normal credential chain. For example:

```bash
aws sso login --profile your-profile
export AWS_PROFILE=your-profile
```

Do not commit `.env` or long-lived AWS access keys. The repository ignores
`.env`, but keeps `.env.example` as the safe configuration template.

Important local options:

| Variable | Purpose |
| --- | --- |
| `INCLUDE_ORIGINAL_CONTENT` | Include original messages in the API response |
| `INCLUDE_LLM_CALL_INPUTS` | Include generated prompts in the response |
| `CORS_ALLOWED_ORIGINS` | Comma-separated browser origins; `*` is local-only |
| `LLM_INPUT_COST_PER_MILLION_TOKENS_USD` | Enables input-cost estimates |
| `LLM_OUTPUT_COST_PER_MILLION_TOKENS_USD` | Enables output-cost estimates |

See [`main/.env.example`](main/.env.example) for every available setting.

## Run the tests

From `main`:

```bash
.venv/bin/python -m pytest
```

The test suite runs without calling AWS.

## Start the API

From `main`:

```bash
.venv/bin/python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8080 \
  --reload
```

Once started:

- Health check: <http://localhost:8080/health>
- Swagger UI: <http://localhost:8080/docs>
- OpenAPI schema: <http://localhost:8080/openapi.json>

Check that the service is running:

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{"status":"ok"}
```

## Summarize a conversation

With `AUTH_ENABLED=false`, send a request without a token:

```bash
curl -X POST http://localhost:8080/api/summarize \
  -H 'Content-Type: application/json' \
  --data '{
    "messages": [
      {
        "role": "customer",
        "content": "My card ending 4242 was declined.",
        "timestamp": "2026-07-30T09:00:00Z",
        "sender": "Customer"
      },
      {
        "role": "support",
        "content": "The block has been removed.",
        "timestamp": "2026-07-30T09:05:00Z",
        "sender": "Agent"
      }
    ],
    "summary_style": "brief",
    "language": "en"
  }'
```

Request rules:

- `messages` must contain between 1 and 1,000 items.
- `role` must be `user`, `assistant`, `system`, `agent`, `customer`, or `support`.
- `content` must not be blank.
- `summary_style` may be `brief`, `detailed`, or `bullet_points`.
- `timestamp` and `sender` are optional.

The response includes the summary, confidence information, sanitized messages,
PII counts, token and cost estimates, and timeline metrics when timestamps are
available.

## Enable authentication

Set `AUTH_ENABLED=true` and configure:

```dotenv
OIDC_JWKS_URL=https://your-idp.example/path/to/jwks
OIDC_ISSUER=https://your-idp.example/issuer
OIDC_AUDIENCE=api://summarizer
OIDC_REQUIRED_SCOPE=summarizer.invoke
OIDC_ALLOWED_ALGORITHMS=RS256
```

Then include an OAuth2 client-credentials token:

```bash
curl -X POST http://localhost:8080/api/summarize \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @request.json
```

`GET /health` remains unauthenticated for load balancer health checks.

## Required AWS permissions

The runtime identity needs:

- `bedrock:ApplyGuardrail` for the configured Guardrail
- `bedrock:InvokeModel` for the configured model or inference profile

Verify local credentials with:

```bash
aws sts get-caller-identity
aws bedrock list-foundation-models --region "$AWS_REGION"
```

If the model uses cross-Region inference, the IAM policy may also need access to
the inference profile and its destination foundation-model resources.

## Production deployment

The `main/deploy` directory contains templates for:

- an EC2 IAM policy
- a systemd service
- Nginx
- log rotation
- the CloudWatch Agent
- CloudWatch alarms

Before running with `APP_ENV=ec2`:

1. Attach an IAM role to the instance; do not configure static AWS credentials.
2. Set `AUTH_ENABLED=true` and provide all required OIDC values.
3. Set `INCLUDE_ORIGINAL_CONTENT=false`.
4. Set `INCLUDE_LLM_CALL_INPUTS=false`.
5. Set `LOG_SANITIZATION_DETAILS=false`.
6. Replace wildcard CORS with explicit origins or disable CORS.
7. Review and replace every placeholder in the deployment templates.

The application refuses to start in EC2 mode when these safety requirements are
not met.

For the complete authentication, IAM, EC2, Nginx, CloudWatch, and alarm setup,
see the [deployment guide](docs/deployment.md).

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Startup reports a missing model or Guardrail | Set all `BEDROCK_*` values in `main/.env` |
| AWS reports missing credentials | Log in through AWS SSO and set `AWS_PROFILE` |
| API returns `401` | Check the token signature, issuer, audience, and expiry |
| API returns `403` | Check that the token contains every required scope or role |
| API returns `429` | Bedrock throttled the request; retry after a delay |
| API returns `502` | Check model access, model ID, Region, and IAM permissions |
| API returns `503` | Check Guardrail configuration, access, and IdP JWKS availability |

Each response includes an `X-Request-ID`, which is also written to application
logs for tracing. Conversation content is not intentionally logged.
