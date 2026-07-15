# Bedrock summarization pipeline

This is the modular replacement for `../code`. It preserves the two-pass summarization flow while routing LLM access through AWS Bedrock instead of a direct third-party API.

## Flow

1. Validate `POST /api/summarize`.
2. Sanitize each message with the independent Amazon Bedrock `ApplyGuardrail` API.
3. Build a role-labelled sanitized conversation.
4. Generate a summary through the provider interface.
5. Verify the summary in a second LLM pass.
6. Combine rule-based confidence (40%) with verifier confidence (60%).
7. Calculate configurable token-cost estimates.
8. Extract timeline metrics and return the existing response shape.

Sanitization is mandatory and fail-closed: if the Guardrail fails, blocks content,
or detects sensitive information without masking it, content is not sent to the
Bedrock foundation model.

## Local setup

```bash
cd Summarizer/main
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Populate `AWS_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_GUARDRAIL_ID`, and
`BEDROCK_GUARDRAIL_VERSION`. Boto3 discovers credentials through its standard
credential chain; application code does not accept a direct provider API key.

Run tests without AWS credentials:

```bash
.venv/bin/python -m pytest
```

Run the service:

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Test using repository sample data:

```bash
curl -X POST http://localhost:8080/api/summarize \
  -H 'Content-Type: application/json' \
  --data @../data/input_from_dtm_pdf.json
```

## AWS access required for a live test

Provide or confirm:

1. AWS region containing the approved Bedrock model.
2. Approved Bedrock model ID or inference-profile ID.
3. A numbered Bedrock Guardrail version configured to mask PII and custom identifiers.
4. An IAM role/profile with `bedrock:ApplyGuardrail` for the Guardrail and
   `bedrock:InvokeModel` for the model resource.
5. Confirmation that Guardrail and model access are enabled in the account and region.
6. Model input/output prices if cost estimates must be populated.

For a developer machine, authenticate using the organisation's normal AWS SSO/profile workflow and export `AWS_PROFILE`. In AWS, use the workload's IAM role. Long-lived access keys should not be placed in `.env` or committed.

Useful credential checks:

```bash
aws sts get-caller-identity
aws bedrock list-foundation-models --region "$AWS_REGION"
```

The Guardrail runs independently before each model call. Raw Guardrail assessment
matches are not logged because they can contain the original PII value.

`INCLUDE_ORIGINAL_CONTENT=true` preserves the current response contract. Set it to `false` where returning original PII is not permitted.
