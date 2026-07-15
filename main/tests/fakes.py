import re

from app.providers.base import LLMResult, TokenUsage
from app.services.guardrail import SanitizationResult


class FakeSanitizer:
    def sanitize(self, text: str) -> SanitizationResult:
        emails = re.findall(r"[\w.+-]+@[\w.-]+", text)
        sanitized = re.sub(r"[\w.+-]+@[\w.-]+", "[EMAIL]", text)
        return SanitizationResult(
            sanitized_text=sanitized,
            original_text=text,
            detected_entities=["EMAIL_ADDRESS"] if emails else [],
            entity_count=len(emails),
        )


class FakeProvider:
    model_id = "test.bedrock-model-v1"

    def __init__(self):
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> LLMResult:
        self.calls.append(kwargs)
        if "QA verifier" in kwargs["system_prompt"]:
            text = (
                '{"verifier_score": 0.9, "reasoning": "Faithful summary", '
                '"missing_points": [], "unsupported_claims": []}'
            )
            return LLMResult(text=text, usage=TokenUsage(80, 20, 100))
        return LLMResult(
            text='{"summary": "The conversation was reviewed and support provided an update."}',
            usage=TokenUsage(100, 25, 125),
        )
