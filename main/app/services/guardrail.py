"""PII sanitization through the independent Amazon Bedrock Guardrail API."""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Protocol

from app.core.config import Settings

logger = logging.getLogger(__name__)


class SanitizationError(RuntimeError):
    """Raised when PII sanitization cannot be guaranteed."""


@dataclass(frozen=True)
class SanitizationResult:
    sanitized_text: str
    original_text: str
    detected_entities: list[str] = field(default_factory=list)
    entity_count: int = 0


class Sanitizer(Protocol):
    def sanitize(self, text: str) -> SanitizationResult: ...


_ENTITY_NAMES = {
    "NAME": "PERSON",
    "EMAIL": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER",
    "ADDRESS": "ADDRESS",
    "UK_NATIONAL_HEALTH_SERVICE_NUMBER": "UK_NHS",
    "UK_NATIONAL_INSURANCE_NUMBER": "UK_NATIONAL_INSURANCE_NUMBER",
    "DRIVER_ID": "DRIVING_LICENSE",
    "LICENSE_PLATE": "VEHICLE_REGISTRATION",
    "CREDIT_DEBIT_CARD_NUMBER": "CREDIT_CARD",
    "INTERNATIONAL_BANK_ACCOUNT_NUMBER": "IBAN_CODE",
    "IP_ADDRESS": "IP_ADDRESS",
    "URL": "URL",
    "AWS_ACCESS_KEY": "AWS_ACCESS_KEY",
    "AWS_SECRET_KEY": "AWS_SECRET_KEY",
    "PASSWORD": "PASSWORD",
    "UK_VEHICLE_REGISTRATION": "VEHICLE_REGISTRATION",
}

_PLACEHOLDERS = {
    "NAME": "PERSON",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "ADDRESS": "ADDRESS",
    "UK_NATIONAL_HEALTH_SERVICE_NUMBER": "NHS_NUMBER",
    "UK_NATIONAL_INSURANCE_NUMBER": "NI_NUMBER",
    "DRIVER_ID": "DRIVING_LICENSE",
    "LICENSE_PLATE": "VEHICLE_REGISTRATION",
    "CREDIT_DEBIT_CARD_NUMBER": "CREDIT_CARD",
    "INTERNATIONAL_BANK_ACCOUNT_NUMBER": "IBAN",
    "UK_VEHICLE_REGISTRATION": "VEHICLE_REGISTRATION",
}


def _normalize_entity_name(value: str) -> str:
    return _ENTITY_NAMES.get(value, value)


def _normalize_placeholders(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_name = match.group(1)
        return f"[{_PLACEHOLDERS.get(raw_name, raw_name)}]"

    normalized = re.sub(r"\{([A-Z][A-Z0-9_]*)\}", replace, text)
    # Built-in and custom filters can detect the same value. Keep one readable marker.
    normalized = normalized.replace("[PHONE][UK_MOBILE]", "[PHONE]")
    normalized = normalized.replace("[UK_MOBILE][PHONE]", "[PHONE]")
    normalized = normalized.replace(
        "[VEHICLE_REGISTRATION][VEHICLE_REGISTRATION]", "[VEHICLE_REGISTRATION]"
    )
    return normalized


class BedrockGuardrailSanitizer:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self._client = client

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
    def _detections(assessments: list[dict[str, Any]]) -> list[dict[str, str]]:
        detections: list[dict[str, str]] = []
        for assessment in assessments:
            policy = assessment.get("sensitiveInformationPolicy", {}) or {}
            for item in policy.get("piiEntities", []) or []:
                detections.append(
                    {
                        "type": str(item.get("type", "PII")),
                        "match": str(item.get("match", "")),
                        "action": str(item.get("action", "")),
                    }
                )
            for item in policy.get("regexes", []) or []:
                detections.append(
                    {
                        "type": str(item.get("name", "CUSTOM_REGEX")),
                        "match": str(item.get("match", "")),
                        "action": str(item.get("action", "")),
                    }
                )
        return detections

    def sanitize(self, text: str) -> SanitizationResult:
        if not text or not text.strip():
            return SanitizationResult(sanitized_text=text, original_text=text)
        try:
            self.settings.validate_for_guardrail()
            response = self._get_client().apply_guardrail(
                guardrailIdentifier=self.settings.bedrock_guardrail_id,
                guardrailVersion=self.settings.bedrock_guardrail_version,
                source="INPUT",
                content=[{"text": {"text": text}}],
            )
            action = str(response.get("action", ""))
            assessments = response.get("assessments", []) or []
            detections = self._detections(assessments)
            blocked = [item for item in detections if item["action"] == "BLOCKED"]
            anonymized = [item for item in detections if item["action"] == "ANONYMIZED"]

            if blocked:
                raise SanitizationError(
                    "Bedrock Guardrail blocked message content; no content was sent to the LLM."
                )

            if action == "NONE":
                if detections:
                    raise SanitizationError(
                        "Bedrock Guardrail detected sensitive content without masking it."
                    )
                return SanitizationResult(sanitized_text=text, original_text=text)

            if action != "GUARDRAIL_INTERVENED" or not anonymized:
                raise SanitizationError("Bedrock Guardrail returned an unexpected sanitization result.")

            outputs = response.get("outputs", response.get("output", [])) or []
            masked_text = "\n".join(
                str(item.get("text", "")) for item in outputs if item.get("text")
            ).strip()
            if not masked_text:
                raise SanitizationError("Bedrock Guardrail did not return masked content.")

            # Deduplicate overlapping built-in/custom detections by their matched value.
            distinct_matches = {
                item["match"].casefold() if item["match"] else f"{item['type']}:{index}"
                for index, item in enumerate(anonymized)
            }
            detected_entities = sorted(
                {_normalize_entity_name(item["type"]) for item in anonymized}
            )
            return SanitizationResult(
                sanitized_text=_normalize_placeholders(masked_text),
                original_text=text,
                detected_entities=detected_entities,
                entity_count=len(distinct_matches),
            )
        except SanitizationError:
            raise
        except Exception as exc:
            logger.error(
                "Bedrock Guardrail sanitization failed (%s); inference has been blocked",
                type(exc).__name__,
                exc_info=True,
            )
            raise SanitizationError(
                "PII sanitization through Bedrock Guardrails failed; no content was sent to the LLM."
            ) from exc
