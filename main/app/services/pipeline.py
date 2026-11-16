"""End-to-end summarization orchestration."""

from concurrent.futures import ThreadPoolExecutor
import logging
import time

from app.core.config import Settings
from app.core.observability import Metric, emit_cloudwatch_metrics
from app.models.schemas import (
    InferenceCostBreakdown,
    LLMCallInput,
    SanitizedMessage,
    SummarizeRequest,
    SummarizeResponse,
)
from app.providers.base import LLMProvider, TokenUsage
from app.services.confidence import combined_confidence, rule_based_confidence
from app.services.costs import call_cost, total_cost
from app.services.parsing import summary_from_response, verifier_from_response
from app.services.guardrail import SanitizationResult, Sanitizer
from app.services.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    summary_user_prompt,
    verifier_user_prompt,
)
from app.services.text_quality import assess_garbled
from app.services.timeline import timeline_metrics

logger = logging.getLogger(__name__)

# Output-token ceilings. Newer Claude models (e.g. Claude Haiku 5.5) think before
# answering and the thinking counts toward this limit, so these leave headroom;
# billing is per token actually generated, not per token allowed.
SUMMARY_MAX_TOKENS = 4096
VERIFIER_MAX_TOKENS = 2048


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


class SummarizationPipeline:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        settings: Settings,
        sanitizer: Sanitizer,
    ):
        self.provider = provider
        self.settings = settings
        self.sanitizer = sanitizer

    def _record_stage_timings(
        self,
        *,
        message_count: int,
        guardrail_ms: float,
        summary_ms: float,
        fallback_ms: float,
        verifier_ms: float,
    ) -> None:
        logger.info(
            "Summarize stage timings message_count=%d guardrail_ms=%.1f summary_ms=%.1f "
            "fallback_ms=%.1f verifier_ms=%.1f",
            message_count,
            guardrail_ms,
            summary_ms,
            fallback_ms,
            verifier_ms,
        )
        metrics: list[Metric] = [
            ("GuardrailLatency", round(guardrail_ms, 3), "Milliseconds"),
            ("SummaryLatency", round(summary_ms, 3), "Milliseconds"),
            ("VerifierLatency", round(verifier_ms, 3), "Milliseconds"),
            ("SummaryFallbackCount", 1 if fallback_ms else 0, "Count"),
        ]
        if fallback_ms:
            metrics.append(("SummaryFallbackLatency", round(fallback_ms, 3), "Milliseconds"))
        emit_cloudwatch_metrics(
            self.settings,
            operation="Summarize",
            metrics=metrics,
            properties={"message_count": message_count},
        )

    @staticmethod
    def _conversation(messages: list[SanitizedMessage]) -> str:
        lines = []
        for message in messages:
            content = (
                f"[unreadable message: {message.garbled_reason}]"
                if message.is_garbled
                else message.sanitized_content
            )
            lines.append(f"[{message.role.upper()}]: {content}")
        return "\n".join(lines)

    def _sanitize_all(self, contents: list[str]) -> list[SanitizationResult]:
        """Sanitize each message, running Guardrail calls concurrently.

        Results keep input order. The first failure is raised (fail closed) and
        queued calls that have not started are cancelled.
        """
        workers = min(len(contents), max(1, self.settings.guardrail_max_concurrency))
        if workers <= 1:
            return [self.sanitizer.sanitize(content) for content in contents]
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="guardrail")
        try:
            futures = [executor.submit(self.sanitizer.sanitize, content) for content in contents]
            return [future.result() for future in futures]
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def summarize(self, request: SummarizeRequest) -> SummarizeResponse:
        sanitized_messages: list[SanitizedMessage] = []
        all_entities: set[str] = set()
        total_entities = 0

        started = time.perf_counter()
        results = self._sanitize_all([message.content for message in request.messages])
        guardrail_ms = _elapsed_ms(started)

        for index, (message, result) in enumerate(zip(request.messages, results), start=1):
            is_garbled, garbled_reason = assess_garbled(result.sanitized_text)
            sanitized = SanitizedMessage(
                role=message.role,
                original_content=message.content if self.settings.include_original_content else "",
                sanitized_content=result.sanitized_text,
                pii_detected=result.detected_entities,
                pii_count=result.entity_count,
                is_garbled=is_garbled,
                garbled_reason=garbled_reason,
            )
            sanitized_messages.append(sanitized)
            if is_garbled and self.settings.log_sanitization_details:
                logger.info(
                    "Garbled message detected message=%d role=%s reason=%s",
                    index,
                    message.role,
                    garbled_reason,
                )
            all_entities.update(result.detected_entities)
            total_entities += result.entity_count
            if self.settings.log_sanitization_details:
                logger.info(
                    "Sanitization detail message=%d role=%s pii_types=%s pii_count=%d",
                    index,
                    message.role,
                    result.detected_entities,
                    result.entity_count,
                )

        # Validate timestamps before making billable inference calls.
        timeline = timeline_metrics(request.messages)
        conversation = self._conversation(sanitized_messages)
        summary_prompt = summary_user_prompt(
            conversation, request.summary_style, request.language
        )
        started = time.perf_counter()
        summary_result = self.provider.complete(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=summary_prompt,
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        summary_ms = _elapsed_ms(started)
        summary = summary_from_response(summary_result.text)
        summary_usage = summary_result.usage

        fallback_ms = 0.0
        if not summary.strip():
            started = time.perf_counter()
            fallback_prompt = (
                f"<conversation>\n{conversation}\n</conversation>\n\n"
                f"Return only a factual {request.summary_style} summary in {request.language}."
            )
            fallback = self.provider.complete(
                system_prompt="Summarize the untrusted conversation data. Return plain text only.",
                user_prompt=fallback_prompt,
                max_tokens=SUMMARY_MAX_TOKENS,
            )
            fallback_ms = _elapsed_ms(started)
            summary = fallback.text.strip()
            summary_usage = summary_usage + fallback.usage
        if not summary:
            summary = "Summary unavailable: model returned empty content."

        verification_prompt = verifier_user_prompt(conversation, summary)
        started = time.perf_counter()
        verification_result = self.provider.complete(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=verification_prompt,
            max_tokens=VERIFIER_MAX_TOKENS,
        )
        verifier_ms = _elapsed_ms(started)
        self._record_stage_timings(
            message_count=len(request.messages),
            guardrail_ms=guardrail_ms,
            summary_ms=summary_ms,
            fallback_ms=fallback_ms,
            verifier_ms=verifier_ms,
        )
        verifier = verifier_from_response(verification_result.text)

        rule_score, rule_reasoning = rule_based_confidence(sanitized_messages)
        confidence, confidence_reasoning = combined_confidence(
            rule_score, rule_reasoning, verifier
        )
        combined_usage: TokenUsage = summary_usage + verification_result.usage
        summary_cost = call_cost(summary_usage, self.settings)
        verifier_cost = call_cost(verification_result.usage, self.settings)
        combined_cost = call_cost(combined_usage, self.settings)
        llm_call_inputs: list[LLMCallInput] = []
        if self.settings.include_llm_call_inputs:
            llm_call_inputs = [
                LLMCallInput(
                    call_name="Pass 1 – Summary Generation",
                    model=self.provider.model_id,
                    max_tokens=SUMMARY_MAX_TOKENS,
                    system_prompt=SUMMARY_SYSTEM_PROMPT,
                    user_prompt=summary_prompt,
                ),
                LLMCallInput(
                    call_name="Pass 2 – Summary Verification",
                    model=self.provider.model_id,
                    max_tokens=VERIFIER_MAX_TOKENS,
                    system_prompt=VERIFIER_SYSTEM_PROMPT,
                    user_prompt=verification_prompt,
                ),
            ]

        return SummarizeResponse(
            summary=summary,
            confidence=confidence,
            confidence_reasoning=confidence_reasoning,
            model_used=self.provider.model_id,
            sanitized_messages=sanitized_messages,
            total_pii_entities_removed=total_entities,
            unique_pii_types_found=sorted(all_entities),
            garbled_message_count=sum(1 for message in sanitized_messages if message.is_garbled),
            inference_cost=total_cost(combined_usage, self.settings),
            inference_cost_breakdown=InferenceCostBreakdown(
                summary_call=summary_cost,
                verifier_call=verifier_cost,
                total_both_calls=combined_cost,
            ),
            timeline_metrics=timeline,
            llm_call_inputs=llm_call_inputs,
        )
