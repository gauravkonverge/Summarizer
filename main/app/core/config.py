"""Environment-backed application configuration."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "AI Summary API")
    aws_region: str = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-2"))
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "")
    bedrock_guardrail_id: str = os.getenv("BEDROCK_GUARDRAIL_ID", "")
    bedrock_guardrail_version: str = os.getenv("BEDROCK_GUARDRAIL_VERSION", "")
    bedrock_endpoint_url: str | None = os.getenv("BEDROCK_ENDPOINT_URL") or None
    llm_max_attempts: int = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
    llm_retry_base_delay_seconds: float = float(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "1.0"))
    llm_retry_max_delay_seconds: float = float(os.getenv("LLM_RETRY_MAX_DELAY_SECONDS", "8.0"))
    log_sanitization_details: bool = _env_bool("LOG_SANITIZATION_DETAILS", False)
    include_original_content: bool = _env_bool("INCLUDE_ORIGINAL_CONTENT", True)
    input_cost_per_million_tokens_usd: float = float(
        os.getenv("LLM_INPUT_COST_PER_MILLION_TOKENS_USD", "0")
    )
    output_cost_per_million_tokens_usd: float = float(
        os.getenv("LLM_OUTPUT_COST_PER_MILLION_TOKENS_USD", "0")
    )
    usd_to_gbp_exchange_rate: float = float(os.getenv("USD_TO_GBP_EXCHANGE_RATE", "0.79"))

    def validate_for_live_inference(self) -> None:
        if not self.bedrock_model_id.strip():
            raise RuntimeError("BEDROCK_MODEL_ID is not configured.")
        if self.llm_max_attempts < 1:
            raise RuntimeError("LLM_MAX_ATTEMPTS must be at least 1.")

    def validate_for_guardrail(self) -> None:
        if not self.bedrock_guardrail_id.strip():
            raise RuntimeError("BEDROCK_GUARDRAIL_ID is not configured.")
        if not self.bedrock_guardrail_version.strip():
            raise RuntimeError("BEDROCK_GUARDRAIL_VERSION is not configured.")
