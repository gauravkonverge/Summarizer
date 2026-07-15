import pytest

from app.core.config import Settings
from app.services.guardrail import BedrockGuardrailSanitizer, SanitizationError


class FakeGuardrailClient:
    def __init__(self, response):
        self.response = response
        self.request = None

    def apply_guardrail(self, **kwargs):
        self.request = kwargs
        return self.response


def settings() -> Settings:
    return Settings(
        bedrock_guardrail_id="guardrail123",
        bedrock_guardrail_version="1",
    )


def test_guardrail_masks_and_normalizes_built_in_and_custom_entities():
    client = FakeGuardrailClient(
        {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [
                {
                    "text": (
                        "My name is {NAME}, email {EMAIL}, phone {PHONE}{UK_MOBILE}, "
                        "registration {UK_VEHICLE_REGISTRATION}, policy {POLICY_ID}."
                    )
                }
            ],
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [
                            {"type": "NAME", "match": "John Smith", "action": "ANONYMIZED"},
                            {"type": "EMAIL", "match": "john@example.com", "action": "ANONYMIZED"},
                            {"type": "PHONE", "match": "07359515942", "action": "ANONYMIZED"},
                        ],
                        "regexes": [
                            {"name": "UK_MOBILE", "match": "07359515942", "action": "ANONYMIZED"},
                            {
                                "name": "UK_VEHICLE_REGISTRATION",
                                "match": "AB12CDE",
                                "action": "ANONYMIZED",
                            },
                            {"name": "POLICY_ID", "match": "POL-123456", "action": "ANONYMIZED"},
                        ],
                    }
                }
            ],
        }
    )
    sanitizer = BedrockGuardrailSanitizer(settings(), client=client)

    result = sanitizer.sanitize("synthetic input")

    assert result.sanitized_text == (
        "My name is [PERSON], email [EMAIL], phone [PHONE], "
        "registration [VEHICLE_REGISTRATION], policy [POLICY_ID]."
    )
    assert result.entity_count == 5
    assert result.detected_entities == [
        "EMAIL_ADDRESS",
        "PERSON",
        "PHONE_NUMBER",
        "POLICY_ID",
        "UK_MOBILE",
        "VEHICLE_REGISTRATION",
    ]
    assert client.request["source"] == "INPUT"
    assert client.request["guardrailIdentifier"] == "guardrail123"


def test_guardrail_returns_original_text_when_no_sensitive_information_is_detected():
    client = FakeGuardrailClient({"action": "NONE", "outputs": [], "assessments": []})
    sanitizer = BedrockGuardrailSanitizer(settings(), client=client)

    result = sanitizer.sanitize("The vehicle is awaiting diagnosis.")

    assert result.sanitized_text == "The vehicle is awaiting diagnosis."
    assert result.entity_count == 0


def test_guardrail_fails_closed_when_content_is_blocked():
    client = FakeGuardrailClient(
        {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [{"text": "Blocked"}],
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [
                            {"type": "PASSWORD", "match": "secret", "action": "BLOCKED"}
                        ]
                    }
                }
            ],
        }
    )

    with pytest.raises(SanitizationError, match="blocked"):
        BedrockGuardrailSanitizer(settings(), client=client).sanitize("synthetic input")
