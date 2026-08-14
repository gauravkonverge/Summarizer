"""Portkey AI Gateway implementation using the OpenAI-compatible chat completions API."""

import logging
import time
from typing import Any

from app.core.config import Settings
from app.core.observability import emit_cloudwatch_metrics
from app.providers.base import LLMProviderError, LLMResult, TokenUsage

logger = logging.getLogger(__name__)


class PortkeyProvider:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self._client = client

    @property
    def model_id(self) -> str:
        return self.settings.llm_model_id

    def _get_client(self):
        if self._client is None:
            from portkey_ai import Portkey

            kwargs: dict[str, Any] = {"api_key": self.settings.portkey_api_key}
            if self.settings.portkey_provider:
                kwargs["provider"] = self.settings.portkey_provider
            if self.settings.portkey_virtual_key:
                kwargs["virtual_key"] = self.settings.portkey_virtual_key
            if self.settings.portkey_base_url:
                kwargs["base_url"] = self.settings.portkey_base_url

            self._client = Portkey(**kwargs)
        return self._client

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        return getattr(exc, "status_code", None)

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
                response = self._get_client().chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = str(response.choices[0].message.content or "").strip()
                usage = response.usage
                return LLMResult(
                    text=text,
                    usage=TokenUsage(
                        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                    ),
                )
            except Exception as exc:
                status_code = self._status_code(exc)
                rate_limited = status_code == 429
                retryable = rate_limited or status_code is None or status_code >= 500
                if retryable and attempt < attempts:
                    delay = min(
                        self.settings.llm_retry_base_delay_seconds * (2 ** (attempt - 1)),
                        self.settings.llm_retry_max_delay_seconds,
                    )
                    logger.warning(
                        "Portkey request failed with status=%s; retrying in %.1fs",
                        status_code,
                        delay,
                    )
                    emit_cloudwatch_metrics(
                        self.settings,
                        operation="Portkey",
                        metrics=[("PortkeyRetry", 1, "Count")],
                        properties={"status_code": str(status_code)},
                    )
                    time.sleep(delay)
                    continue
                logger.error("Portkey request failed with status=%s", status_code, exc_info=True)
                raise LLMProviderError(
                    "Portkey LLM gateway request failed.",
                    rate_limited=rate_limited,
                ) from exc

        raise LLMProviderError("Portkey LLM gateway request failed.")
