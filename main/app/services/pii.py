"""Fail-closed PII detection using Presidio plus UK-oriented recognizers."""

from dataclasses import dataclass, field
import importlib.util
import logging
import os
import re
from typing import Iterable

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)

SUPPORTED_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "LOCATION",
    "URL",
    "IBAN_CODE",
    "ORGANIZATION",
]

ENTITY_LABELS = {
    "PERSON": "[PERSON]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "UK_MOBILE": "[UK_MOBILE]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "UK_NHS": "[NHS_NUMBER]",
    "IP_ADDRESS": "[IP_ADDRESS]",
    "LOCATION": "[LOCATION]",
    "URL": "[URL]",
    "IBAN_CODE": "[IBAN]",
    "ORGANIZATION": "[ORGANIZATION]",
    "ADDRESS": "[ADDRESS]",
    "SSN": "[SSN]",
    "DRIVING_LICENSE": "[DRIVING_LICENSE]",
    "VEHICLE_REGISTRATION": "[VEHICLE_REGISTRATION]",
    "POLICY_ID": "[POLICY_ID]",
}


class SanitizationError(RuntimeError):
    """Raised when sanitization cannot be guaranteed."""


@dataclass(frozen=True)
class SanitizationResult:
    sanitized_text: str
    original_text: str
    detected_entities: list[str] = field(default_factory=list)
    entity_count: int = 0


class PIISanitizer:
    _SIMPLE_PATTERNS = {
        "UK_MOBILE": [
            r"\b07\d{3}\s?\d{3}\s?\d{3}\b",
            r"\+447\d{3}\s?\d{3}\s?\d{3}\b",
            r"0044\s?7\d{3}\s?\d{3}\s?\d{3}\b",
        ],
        "ADDRESS": [
            r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Za-z0-9'.,-]+\s+){0,5}"
            r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Close|Court|Ct|Way)\b"
        ],
        "SSN": [r"\b\d{3}-\d{2}-\d{4}\b"],
        "VEHICLE_REGISTRATION": [r"\b[A-Z]{2}\d{2}[A-Z]{3}\b"],
    }
    _CONTEXT_PATTERNS = {
        "UK_NHS": [r"(?i)\bNHS(?:\s+number)?\s*(?:is|:)?\s*(\d{3}\s?\d{3}\s?\d{4})\b"],
        "DRIVING_LICENSE": [
            r"(?i)\bdriving\s+licen[cs]e(?:\s+number)?\s*(?:is|:)?\s*([A-Z0-9-]{6,20})\b"
        ],
        "PHONE_NUMBER": [
            r"(?i)\b(?:phone|mobile|telephone|contact(?:\s+number)?|call(?:\s+me)?(?:\s+on)?)"
            r"\s*(?:is|:)?\s*(\+?\d[\d\s().-]{7,}\d)\b"
        ],
        "POLICY_ID": [
            r"(?i)\bpolicy\s*(?:id|number|no\.?|#)?\s*(?:is|:)?\s*([A-Z]{2,6}-\d{3,12})\b",
            r"(?i)\bpolicy\s*(?:id|number|no\.?|#)?\s*(?:is|:)?\s*([A-Z0-9]{6,20})\b",
        ],
    }

    def __init__(self, language: str = "en", confidence_threshold: float = 0.8):
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.spacy_model_name = os.getenv("SPACY_MODEL_NAME", "en_core_web_lg")
        self._analyzer: AnalyzerEngine | None = None
        self._anonymizer: AnonymizerEngine | None = None
        self._simple = {
            entity: [re.compile(pattern, re.IGNORECASE if entity == "ADDRESS" else 0) for pattern in patterns]
            for entity, patterns in self._SIMPLE_PATTERNS.items()
        }
        self._context = {
            entity: [re.compile(pattern) for pattern in patterns]
            for entity, patterns in self._CONTEXT_PATTERNS.items()
        }

    def _engines(self) -> tuple[AnalyzerEngine, AnonymizerEngine]:
        if self._analyzer is None:
            # Presidio otherwise attempts an implicit network download. Production
            # images must contain the approved NLP model ahead of time.
            if importlib.util.find_spec(self.spacy_model_name) is None:
                raise SanitizationError(
                    f"Configured spaCy model '{self.spacy_model_name}' is not installed; "
                    "automatic model downloads are disabled."
                )
            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [
                        {"lang_code": self.language, "model_name": self.spacy_model_name}
                    ],
                }
            )
            self._analyzer = AnalyzerEngine(
                nlp_engine=provider.create_engine(),
                supported_languages=[self.language],
            )
        if self._anonymizer is None:
            self._anonymizer = AnonymizerEngine()
        return self._analyzer, self._anonymizer

    @staticmethod
    def _recognizer(entity: str, start: int, end: int) -> RecognizerResult:
        return RecognizerResult(entity_type=entity, start=start, end=end, score=0.95)

    def _custom_results(self, text: str) -> Iterable[RecognizerResult]:
        for entity, patterns in self._simple.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    yield self._recognizer(entity, match.start(), match.end())
        for entity, patterns in self._context.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    if match.lastindex:
                        yield self._recognizer(entity, match.start(1), match.end(1))

    def sanitize(self, text: str) -> SanitizationResult:
        if not text or not text.strip():
            return SanitizationResult(sanitized_text=text, original_text=text)
        try:
            analyzer, anonymizer = self._engines()
            results = analyzer.analyze(
                text=text,
                entities=SUPPORTED_ENTITIES,
                language=self.language,
            )
            results.extend(self._custom_results(text))
            filtered = [result for result in results if result.score >= self.confidence_threshold]
            # Avoid double-counting identical recognizer results while leaving Presidio to resolve overlaps.
            unique = {
                (result.entity_type, result.start, result.end): result for result in filtered
            }
            filtered = list(unique.values())
            if not filtered:
                return SanitizationResult(sanitized_text=text, original_text=text)
            operators = {
                entity: OperatorConfig("replace", {"new_value": label})
                for entity, label in ENTITY_LABELS.items()
            }
            anonymized = anonymizer.anonymize(
                text=text,
                analyzer_results=filtered,
                operators=operators,
            )
            return SanitizationResult(
                sanitized_text=anonymized.text,
                original_text=text,
                detected_entities=sorted({result.entity_type for result in filtered}),
                entity_count=len(filtered),
            )
        except Exception as exc:
            logger.error("PII sanitization failed; inference has been blocked", exc_info=True)
            raise SanitizationError("PII sanitization failed; no content was sent to the LLM.") from exc
