"""AWS Bedrock implementation using the provider-neutral Converse API."""

import logging
import time
from typing import Any

from app.core.config import Settings
from app.providers.base import LLMProviderError, LLMResult, TokenUsage

logger = logging.getLogger(__name__)

_RATE_LIMIT_CODES = {"ThrottlingException", "TooManyRequestsException", "LimitExceededException"}
_RETRYABLE_CODES = _RATE_LIMIT_CODES | {"ServiceUnavailableException", "InternalServerException"}


class BedrockProvider:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self._client = client

    @property
    def model_id(self) -> str:
        return self.settings.bedrock_model_id

    def _get_client(self):
        if self._client is None:
            import boto3

            kwargs: dict[str, Any] = {
                "service_name": "bedrock-runtime",
                "region_name": self.settings.aws_region,
            }
            if self.settings.bedrock_endpoint_url:
                kwargs["endpoint_url"] = self.settings.bedrock_endpoint_url
            self._client = boto3.client(**kwargs)
        return self._client

    @staticmethod
    def _error_code(exc: Exception) -> str:
        response = getattr(exc, "response", {}) or {}
        return str(response.get("Error", {}).get("Code", ""))

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        self.settings.validate_for_live_inference()
        attempts = max(1, self.settings.llm_max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                response = self._get_client().converse(
                    modelId=self.model_id,
                    system=[{"text": system_prompt}],
                    messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                    inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
                )
                blocks = response.get("output", {}).get("message", {}).get("content", [])
                text = "\n".join(
                    str(block["text"]) for block in blocks if isinstance(block, dict) and block.get("text")
                ).strip()
                usage = response.get("usage", {}) or {}
                return LLMResult(
                    text=text,
                    usage=TokenUsage(
                        input_tokens=int(usage.get("inputTokens", 0)),
                        output_tokens=int(usage.get("outputTokens", 0)),
                        total_tokens=int(usage.get("totalTokens", 0)),
                    ),
                )
            except Exception as exc:
                code = self._error_code(exc)
                retryable = code in _RETRYABLE_CODES
                if retryable and attempt < attempts:
                    delay = min(
                        self.settings.llm_retry_base_delay_seconds * (2 ** (attempt - 1)),
                        self.settings.llm_retry_max_delay_seconds,
                    )
                    logger.warning("Bedrock request failed with %s; retrying in %.1fs", code, delay)
                    time.sleep(delay)
                    continue
                logger.error("Bedrock request failed with code=%s", code or "unknown", exc_info=True)
                raise LLMProviderError(
                    "AWS Bedrock inference request failed.",
                    rate_limited=code in _RATE_LIMIT_CODES,
                ) from exc

        raise LLMProviderError("AWS Bedrock inference request failed.")
