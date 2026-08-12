"""Environment-backed application configuration."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# Symmetric algorithms must never be paired with IdP-published verification
# keys; accepting one enables an algorithm-confusion forgery.
_ASYMMETRIC_ALGORITHM_PREFIXES = ("RS", "PS", "ES", "Ed")


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "local").strip().lower()
    app_name: str = os.getenv("APP_NAME", "AI Summary API")
    cloudwatch_environment: str = os.getenv(
        "CLOUDWATCH_ENVIRONMENT",
        os.getenv("APP_ENV", "local").strip().lower(),
    )
    aws_region: str = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-2"))
    bedrock_guardrail_id: str = os.getenv("BEDROCK_GUARDRAIL_ID", "")
    bedrock_guardrail_version: str = os.getenv("BEDROCK_GUARDRAIL_VERSION", "")
    bedrock_endpoint_url: str | None = os.getenv("BEDROCK_ENDPOINT_URL") or None
    llm_model_id: str = os.getenv("LLM_MODEL_ID", "")
    portkey_api_key: str = os.getenv("PORTKEY_API_KEY", "")
    portkey_provider: str = os.getenv("PORTKEY_PROVIDER", "")
    portkey_virtual_key: str = os.getenv("PORTKEY_VIRTUAL_KEY", "")
    portkey_base_url: str | None = os.getenv("PORTKEY_BASE_URL") or None
    llm_max_attempts: int = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
    llm_retry_base_delay_seconds: float = float(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "1.0"))
    llm_retry_max_delay_seconds: float = float(os.getenv("LLM_RETRY_MAX_DELAY_SECONDS", "8.0"))
    cloudwatch_emf_enabled: bool = _env_bool("CLOUDWATCH_EMF_ENABLED", False)
    cloudwatch_metrics_namespace: str = os.getenv(
        "CLOUDWATCH_METRICS_NAMESPACE", "Corpay/Summarizer"
    )
    log_sanitization_details: bool = _env_bool("LOG_SANITIZATION_DETAILS", False)
    include_original_content: bool = _env_bool("INCLUDE_ORIGINAL_CONTENT", True)
    include_llm_call_inputs: bool = _env_bool("INCLUDE_LLM_CALL_INPUTS", True)
    input_cost_per_million_tokens_usd: float = float(
        os.getenv("LLM_INPUT_COST_PER_MILLION_TOKENS_USD", "0")
    )
    output_cost_per_million_tokens_usd: float = float(
        os.getenv("LLM_OUTPUT_COST_PER_MILLION_TOKENS_USD", "0")
    )
    usd_to_gbp_exchange_rate: float = float(os.getenv("USD_TO_GBP_EXCHANGE_RATE", "0.79"))
    auth_enabled: bool = _env_bool("AUTH_ENABLED", False)
    oidc_jwks_url: str = os.getenv("OIDC_JWKS_URL", "")
    oidc_issuer: str = os.getenv("OIDC_ISSUER", "")
    oidc_audience: str = os.getenv("OIDC_AUDIENCE", "")
    oidc_required_scope: str = os.getenv("OIDC_REQUIRED_SCOPE", "")
    oidc_allowed_algorithms: str = os.getenv("OIDC_ALLOWED_ALGORITHMS", "RS256")
    oidc_jwks_cache_seconds: int = int(os.getenv("OIDC_JWKS_CACHE_SECONDS", "300"))
    oidc_clock_skew_seconds: int = int(os.getenv("OIDC_CLOCK_SKEW_SECONDS", "60"))
    cors_allowed_origins: str = os.getenv("CORS_ALLOWED_ORIGINS", "*")

    def oidc_allowed_algorithm_list(self) -> list[str]:
        return _csv(self.oidc_allowed_algorithms)

    def oidc_required_scope_list(self) -> set[str]:
        return set(_csv(self.oidc_required_scope))

    def cors_allowed_origin_list(self) -> list[str]:
        return _csv(self.cors_allowed_origins)

    def validate_runtime(self) -> None:
        runtime = self.app_env.strip().lower()
        if runtime not in {"local", "ec2"}:
            raise RuntimeError("APP_ENV must be either 'local' or 'ec2'.")
        if self.cloudwatch_emf_enabled:
            if not self.cloudwatch_metrics_namespace.strip():
                raise RuntimeError(
                    "CLOUDWATCH_METRICS_NAMESPACE must not be blank when CloudWatch EMF is enabled."
                )
            if not self.cloudwatch_environment.strip():
                raise RuntimeError(
                    "CLOUDWATCH_ENVIRONMENT must not be blank when CloudWatch EMF is enabled."
                )

        if runtime != "ec2":
            return

        local_credential_variables = (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
        )
        configured = [name for name in local_credential_variables if os.getenv(name)]
        if configured:
            raise RuntimeError(
                "Local AWS credentials must not be configured when APP_ENV=ec2; "
                "attach an IAM role to the EC2 instance instead."
            )
        if self.include_original_content:
            raise RuntimeError("INCLUDE_ORIGINAL_CONTENT must be false when APP_ENV=ec2.")
        if self.include_llm_call_inputs:
            raise RuntimeError("INCLUDE_LLM_CALL_INPUTS must be false when APP_ENV=ec2.")
        if self.log_sanitization_details:
            raise RuntimeError("LOG_SANITIZATION_DETAILS must be false when APP_ENV=ec2.")
        # if not self.auth_enabled:
        #     raise RuntimeError("AUTH_ENABLED must be true when APP_ENV=ec2.")
        if "*" in self.cors_allowed_origin_list():
            raise RuntimeError(
                "CORS_ALLOWED_ORIGINS must not be '*' when APP_ENV=ec2; "
                "list explicit origins or leave it empty to disable CORS."
            )

    def validate_for_auth(self) -> None:
        if not self.auth_enabled:
            return
        if not self.oidc_jwks_url.strip():
            raise RuntimeError("OIDC_JWKS_URL is not configured but AUTH_ENABLED is true.")
        if not self.oidc_issuer.strip():
            raise RuntimeError("OIDC_ISSUER is not configured but AUTH_ENABLED is true.")
        if not self.oidc_audience.strip():
            raise RuntimeError("OIDC_AUDIENCE is not configured but AUTH_ENABLED is true.")
        algorithms = self.oidc_allowed_algorithm_list()
        if not algorithms:
            raise RuntimeError("OIDC_ALLOWED_ALGORITHMS must list at least one algorithm.")
        unsupported = [
            algorithm
            for algorithm in algorithms
            if not algorithm.startswith(_ASYMMETRIC_ALGORITHM_PREFIXES)
        ]
        if unsupported:
            raise RuntimeError(
                "OIDC_ALLOWED_ALGORITHMS must contain only asymmetric algorithms "
                f"(RS*/PS*/ES*/Ed*); rejected: {', '.join(unsupported)}."
            )
        if self.oidc_jwks_cache_seconds < 1:
            raise RuntimeError("OIDC_JWKS_CACHE_SECONDS must be at least 1.")
        if self.oidc_clock_skew_seconds < 0:
            raise RuntimeError("OIDC_CLOCK_SKEW_SECONDS must not be negative.")

    def validate_for_live_inference(self) -> None:
        if not self.portkey_api_key.strip():
            raise RuntimeError("PORTKEY_API_KEY is not configured.")
        if not self.portkey_provider.strip() and not self.portkey_virtual_key.strip():
            raise RuntimeError("PORTKEY_PROVIDER or PORTKEY_VIRTUAL_KEY must be configured.")
        if not self.llm_model_id.strip():
            raise RuntimeError("LLM_MODEL_ID is not configured.")
        if self.llm_max_attempts < 1:
            raise RuntimeError("LLM_MAX_ATTEMPTS must be at least 1.")

    def validate_for_guardrail(self) -> None:
        if not self.bedrock_guardrail_id.strip():
            raise RuntimeError("BEDROCK_GUARDRAIL_ID is not configured.")
        if not self.bedrock_guardrail_version.strip():
            raise RuntimeError("BEDROCK_GUARDRAIL_VERSION is not configured.")
