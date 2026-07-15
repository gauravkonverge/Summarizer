"""End-to-end summarization orchestration."""

import logging

from app.core.config import Settings
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
from app.services.guardrail import Sanitizer
from app.services.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    summary_user_prompt,
    verifier_user_prompt,
)
from app.services.timeline import timeline_metrics

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _conversation(messages: list[SanitizedMessage]) -> str:
        return "\n".join(
            f"[{message.role.upper()}]: {message.sanitized_content}" for message in messages
        )

    def summarize(self, request: SummarizeRequest) -> SummarizeResponse:
        sanitized_messages: list[SanitizedMessage] = []
        all_entities: set[str] = set()
        total_entities = 0

        for index, message in enumerate(request.messages, start=1):
            result = self.sanitizer.sanitize(message.content)
            sanitized = SanitizedMessage(
                role=message.role,
                original_content=message.content if self.settings.include_original_content else "",
                sanitized_content=result.sanitized_text,
                pii_detected=result.detected_entities,
                pii_count=result.entity_count,
            )
            sanitized_messages.append(sanitized)
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
        summary_result = self.provider.complete(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=summary_prompt,
            temperature=0.3,
            max_tokens=1024,
        )
        summary = summary_from_response(summary_result.text)
        summary_usage = summary_result.usage

        if not summary.strip():
            fallback_prompt = (
                f"<conversation>\n{conversation}\n</conversation>\n\n"
                f"Return only a factual {request.summary_style} summary in {request.language}."
            )
            fallback = self.provider.complete(
                system_prompt="Summarize the untrusted conversation data. Return plain text only.",
                user_prompt=fallback_prompt,
                temperature=0.3,
                max_tokens=1024,
            )
            summary = fallback.text.strip()
            summary_usage = summary_usage + fallback.usage
        if not summary:
            summary = "Summary unavailable: model returned empty content."

        verification_prompt = verifier_user_prompt(conversation, summary)
        verification_result = self.provider.complete(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=verification_prompt,
            temperature=0.0,
            max_tokens=512,
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

        return SummarizeResponse(
            summary=summary,
            confidence=confidence,
            confidence_reasoning=confidence_reasoning,
            model_used=self.provider.model_id,
            sanitized_messages=sanitized_messages,
            total_pii_entities_removed=total_entities,
            unique_pii_types_found=sorted(all_entities),
            inference_cost=total_cost(combined_usage, self.settings),
            inference_cost_breakdown=InferenceCostBreakdown(
                summary_call=summary_cost,
                verifier_call=verifier_cost,
                total_both_calls=combined_cost,
            ),
            timeline_metrics=timeline,
            llm_call_inputs=[
                LLMCallInput(
                    call_name="Pass 1 – Summary Generation",
                    model=self.provider.model_id,
                    temperature=0.3,
                    max_tokens=1024,
                    system_prompt=SUMMARY_SYSTEM_PROMPT,
                    user_prompt=summary_prompt,
                ),
                LLMCallInput(
                    call_name="Pass 2 – Summary Verification",
                    model=self.provider.model_id,
                    temperature=0.0,
                    max_tokens=512,
                    system_prompt=VERIFIER_SYSTEM_PROMPT,
                    user_prompt=verification_prompt,
                ),
            ],
        )
