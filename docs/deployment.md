# Bedrock Summarizer Deployment Guide

This is the modular replacement for `../code`. It preserves the two-pass summarization flow while routing LLM access through AWS Bedrock instead of a direct third-party API.

## Flow

0. Authenticate the caller (see [Authentication](#authentication)).
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

## Authentication

`/api/summarize` requires an OAuth2 client-credentials access token issued by the
organisation's IdP (Entra ID, Okta, or Ping):

```
Authorization: Bearer <access token>
```

The token signature is verified against the IdP's published JWKS document, and
`iss`, `aud`, `exp`, and `iat` are all required. Only asymmetric signing
algorithms are accepted; configuring `HS*` is rejected at startup because a
symmetric algorithm combined with a published verification key allows forgery.
When `OIDC_REQUIRED_SCOPE` is set, the caller must also hold every listed scope.
`scope`, `scp`, and `roles` claims are all read, which covers the three IdPs
above without per-vendor configuration.

This service stores no caller secrets. Token issuance, expiry, rotation, and
revocation stay with the IdP.

| Result | Status |
| --- | --- |
| Valid token with required scope | `200` |
| Missing, malformed, expired, or wrongly-signed token | `401` |
| Valid token lacking a required scope | `403` |
| IdP JWKS endpoint unreachable | `503` |

`GET /health` is intentionally unauthenticated so load balancer health checks
work without credentials. It returns no conversation data.

### Onboarding a calling application

1. Register the API in the IdP as its own application, and set `OIDC_AUDIENCE`
   to the resulting identifier (for example `api://summarizer`).
2. Expose an application permission or scope named to match
   `OIDC_REQUIRED_SCOPE`, for example `summarizer.invoke`.
3. Register the calling application, grant it that permission, and have it
   request a token using the client-credentials grant.
4. Set `OIDC_ISSUER` to the exact `iss` value in issued tokens and
   `OIDC_JWKS_URL` to the IdP's key endpoint. Both are non-secret.
5. Confirm the instance security group only admits the caller.

Callers should cache the token until shortly before `exp` rather than requesting
one per call.

### Why not API Gateway with IAM SigV4

An AWS-native alternative would be API Gateway in front of this service using
IAM SigV4, which removes shared secrets entirely. It is not used because the
API Gateway integration timeout is capped at 29–30 seconds and cannot be
raised. A single request performs one `ApplyGuardrail` call per message plus two
or three model calls, which exceeds that ceiling for long conversations; the
supplied Nginx template allows 120 seconds for the same reason. Authenticating
in the application avoids the ceiling and works identically behind an
Application Load Balancer, Nginx, or a direct connection.

Rate limiting is not implemented. Each request costs several billable Bedrock
calls, so throttling per caller at the load balancer or via AWS WAF is
worthwhile.

## Common environment configuration

The application has one environment-variable contract for both local development
and EC2. Copy `.env.example` locally, or install the same template as
`/etc/summarizer-api.env` on EC2.

`APP_ENV` selects runtime safety rules; it does not select a different pipeline:

| Setting | Local | EC2 |
| --- | --- | --- |
| `APP_ENV` | `local` | `ec2` |
| AWS authentication | temporary keys or `AWS_PROFILE` | attached EC2 IAM role |
| `INCLUDE_ORIGINAL_CONTENT` | configurable | must be `false` |
| `INCLUDE_LLM_CALL_INPUTS` | configurable | must be `false` |
| `LOG_SANITIZATION_DETAILS` | configurable | must be `false` |
| `AUTH_ENABLED` | configurable | must be `true` |
| `CORS_ALLOWED_ORIGINS` | configurable | must not be `*` |

Both modes use the same Bedrock provider, Guardrail sanitizer, and summarization
pipeline. Boto3 selects the credential source through its standard credential chain.

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

Test the endpoint. With `AUTH_ENABLED=false` no token is needed:

```bash
curl -X POST http://localhost:8080/api/summarize \
  -H 'Content-Type: application/json' \
  --data '{"messages":[
    {"role":"customer","content":"My card ending 4242 was declined."},
    {"role":"support","content":"The block has been removed."}
  ],"summary_style":"brief"}'
```

With `AUTH_ENABLED=true`, add the token:

```bash
curl -X POST http://localhost:8080/api/summarize \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @request.json
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

## EC2 deployment

The `deploy/` folder contains templates for the instance role, systemd, and Nginx.
The templates contain placeholders and must be reviewed for the target AWS account,
Region, approved inference profile, and Guardrail.

1. Create an EC2 IAM role from `deploy/iam-policy.json`, replace all placeholders,
   and attach the role to the instance. Cross-Region inference profiles can require
   both the inference-profile ARN and destination foundation-model ARNs.
2. Install this `main` directory at `/opt/summarizer-api`, create the service user,
   and create the virtual environment:

   ```bash
   sudo useradd --system --home /opt/summarizer-api --shell /usr/sbin/nologin summarizer
   sudo chown -R summarizer:summarizer /opt/summarizer-api
   sudo -u summarizer python3 -m venv /opt/summarizer-api/.venv
   sudo -u summarizer /opt/summarizer-api/.venv/bin/python -m pip install -r /opt/summarizer-api/requirements.txt
   ```

3. Copy the common environment template outside the repository:

   ```bash
   sudo cp .env.example /etc/summarizer-api.env
   sudo chmod 600 /etc/summarizer-api.env
   ```

   Set `APP_ENV=ec2`, remove local AWS credential/profile values, populate the
   Bedrock values, set the three sensitive observability options to `false`, set
   `AUTH_ENABLED=true` with the four `OIDC_*` values, and replace
   `CORS_ALLOWED_ORIGINS=*` with explicit origins or an empty value.

4. Install and start the systemd unit:

   ```bash
   sudo cp deploy/summarizer-api.service /etc/systemd/system/
   sudo cp deploy/summarizer-api.logrotate /etc/logrotate.d/summarizer-api
   sudo systemctl daemon-reload
   sudo systemctl enable --now summarizer-api
   sudo systemctl status summarizer-api
   ```

5. Optionally install `deploy/nginx.conf` as the Nginx site configuration. Terminate
   HTTPS at an Application Load Balancer or configure an approved TLS certificate
   in Nginx. Restrict port `8080` to the instance; Uvicorn binds to `127.0.0.1` in
   the supplied systemd unit.

6. Install the Amazon CloudWatch Agent using the EC2 console, Systems Manager,
   or the package for the instance operating system. The instance role in
   `deploy/iam-policy.json` includes the CloudWatch permissions required by the
   agent. Then load the supplied configuration:

   ```bash
   sudo cp deploy/cloudwatch-agent.json \
     /opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-agent.json
   sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
     -a fetch-config -m ec2 -s \
     -c file:/opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-agent.json
   sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
     -a status -m ec2
   ```

   Set `CLOUDWATCH_EMF_ENABLED=true` and a stable deployment name such as
   `CLOUDWATCH_ENVIRONMENT=prod` in `/etc/summarizer-api.env`. Application,
   Nginx access, and Nginx error logs are sent to `/corpay/summarizer/*` log
   groups with 30-day retention. Host and application metrics use the
   `Corpay/Summarizer` namespace. If `CLOUDWATCH_METRICS_NAMESPACE` is changed,
   update the agent configuration's metrics namespace to match.

   The application emits bounded dimensions (`Environment` and `Operation`)
   for request count, latency, HTTP errors, successful summaries, token usage,
   detected PII count, Guardrail failures, Bedrock failures, throttles, and
   retries. Request IDs are returned in `X-Request-ID` and included only as log
   properties, never as metric dimensions.

   Deploy the initial alarms, passing the same environment name and the EC2
   instance ID and the SNS topic that should receive notifications:

   ```bash
   aws cloudformation deploy \
     --stack-name summarizer-cloudwatch-alarms \
     --template-file deploy/cloudwatch-alarms.yaml \
     --parameter-overrides \
       Environment=prod \
       InstanceId=i-0123456789abcdef0 \
       AlarmTopicArn=arn:aws:sns:AWS_REGION:AWS_ACCOUNT_ID:operations
   ```

   The template alarms on API 5xx responses, Bedrock throttles, Guardrail
   failures, p95 request latency, memory/disk use, and EC2 status checks. Tune
   the supplied thresholds against real traffic before paging.

The application will fail at startup when `APP_ENV=ec2` is combined with static AWS
credentials, original-content responses, prompt responses, detailed sanitization
logging, disabled authentication, or wildcard CORS. Incomplete `OIDC_*`
configuration also fails at startup whenever `AUTH_ENABLED=true`. Model and
Guardrail IDs continue to be validated before their respective AWS calls.
