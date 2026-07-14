"""
PII Sanitizer module using Microsoft Presidio.
Detects and anonymizes personally identifiable information from text.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)

# PII confidence threshold: only detections >= this score are considered PII
# Higher threshold reduces false positives (e.g. booking refs flagged as DL)
# Default 0.8 is strict; set to 0.5 for legacy behavior
PII_CONFIDENCE_THRESHOLD = float(os.getenv("PII_CONFIDENCE_THRESHOLD", "0.8"))

# PII entity types to detect and anonymize (UK-focused)
# NOTE: DATE_TIME is intentionally excluded - dates/timeframes are business context,
# not PII, and are critical for incident timeline and customer expectation visibility.
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

# Replacement labels used when anonymizing (UK-focused)
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


@dataclass
class SanitizationResult:
    sanitized_text: str
    original_text: str
    detected_entities: list[str] = field(default_factory=list)
    entity_count: int = 0


class PIISanitizer:
    """
    Wraps Presidio Analyzer + Anonymizer to detect and replace PII in text.
    Includes UK mobile number detection via regex post-processing.
    """

    # UK mobile number patterns
    UK_MOBILE_PATTERNS = [
        r"\b07\d{3}\s?\d{3}\s?\d{3}\b",  # 07xxx xxx xxx
        r"\+447\d{3}\s?\d{3}\s?\d{3}\b",  # +447xxx xxx xxx
        r"0044\s?7\d{3}\s?\d{3}\s?\d{3}\b",  # 0044 7xxx xxx xxx
    ]

    ADDRESS_PATTERNS = [
        (
            r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Za-z0-9'.,-]+\s+){0,5}"
            r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Close|Court|Ct|Way)\b"
        ),
    ]

    PHONE_CONTEXT_PATTERNS = [
        (
            r"(?i)\b(?:phone|mobile|contact(?:\s+number)?|call(?:\s+me)?(?:\s+on)?)"
            r"\s*(?:is|:)?\s*(\+?\d[\d\s().-]{7,}\d)\b"
        ),
    ]

    SSN_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",
    ]

    UK_NHS_CONTEXT_PATTERNS = [
        r"(?i)\bNHS(?:\s+number)?\s*(?:is|:)?\s*(\d{3}\s?\d{3}\s?\d{4})\b",
    ]

    DRIVING_LICENSE_CONTEXT_PATTERNS = [
        r"(?i)\bdriving\s+licen[cs]e(?:\s+number)?\s*(?:is|:)?\s*([A-Z0-9-]{6,20})\b",
    ]

    VEHICLE_REGISTRATION_PATTERNS = [
        r"\b[A-Z]{2}\d{2}[A-Z]{3}\b",
    ]

    POLICY_ID_CONTEXT_PATTERNS = [
        r"(?i)\bpolicy\s*(?:id|number|no\.?|#)?\s*(?:is|:)?\s*([A-Z]{2,6}-\d{3,12})\b",
        r"(?i)\bpolicy\s*(?:id|number|no\.?|#)?\s*(?:is|:)?\s*([A-Z0-9]{6,20})\b",
    ]

    def __init__(self, language: str = "en"):
        self.language = language
        self._analyzer: Optional[AnalyzerEngine] = None
        self._anonymizer: Optional[AnonymizerEngine] = None
        self._uk_mobile_regex = re.compile("|".join(self.UK_MOBILE_PATTERNS))
        self._address_regex = re.compile("|".join(self.ADDRESS_PATTERNS), re.IGNORECASE)
        self._ssn_regex = re.compile("|".join(self.SSN_PATTERNS))
        self._vehicle_registration_regex = re.compile(
            "|".join(self.VEHICLE_REGISTRATION_PATTERNS)
        )
        self._nhs_context_regexes = [
            re.compile(pattern) for pattern in self.UK_NHS_CONTEXT_PATTERNS
        ]
        self._driving_license_context_regexes = [
            re.compile(pattern) for pattern in self.DRIVING_LICENSE_CONTEXT_PATTERNS
        ]
        self._phone_context_regexes = [
            re.compile(pattern) for pattern in self.PHONE_CONTEXT_PATTERNS
        ]
        self._policy_id_context_regexes = [
            re.compile(pattern) for pattern in self.POLICY_ID_CONTEXT_PATTERNS
        ]

    def _get_analyzer(self) -> AnalyzerEngine:
        if self._analyzer is None:
            self._analyzer = AnalyzerEngine()
        return self._analyzer

    def _get_anonymizer(self) -> AnonymizerEngine:
        if self._anonymizer is None:
            self._anonymizer = AnonymizerEngine()
        return self._anonymizer

    def _build_operators(self) -> dict:
        """Build replace operators for each entity type."""
        operators = {}
        for entity, label in ENTITY_LABELS.items():
            operators[entity] = OperatorConfig("replace", {"new_value": label})
        return operators

    def sanitize(self, text: str) -> SanitizationResult:
        """
        Analyze text for PII, anonymize it, and return a SanitizationResult.
        Includes UK mobile number detection via regex.
        """
        if not text or not text.strip():
            return SanitizationResult(sanitized_text=text, original_text=text)

        try:
            analyzer = self._get_analyzer()
            anonymizer = self._get_anonymizer()

            results: list[RecognizerResult] = analyzer.analyze(
                text=text,
                entities=SUPPORTED_ENTITIES,
                language=self.language,
            )

            # Add UK mobile detections via regex post-processing
            for match in self._uk_mobile_regex.finditer(text):
                # Create a synthetic RecognizerResult for UK mobile
                uk_mobile_result = RecognizerResult(
                    entity_type="UK_MOBILE",
                    start=match.start(),
                    end=match.end(),
                    score=0.95,  # High confidence for regex match
                    analysis_explanation=None,
                    recognition_metadata=None,
                )
                results.append(uk_mobile_result)

            # Add address detections via regex post-processing
            for match in self._address_regex.finditer(text):
                address_result = RecognizerResult(
                    entity_type="ADDRESS",
                    start=match.start(),
                    end=match.end(),
                    score=0.95,
                    analysis_explanation=None,
                    recognition_metadata=None,
                )
                results.append(address_result)

            # Add SSN detections via regex post-processing
            for match in self._ssn_regex.finditer(text):
                ssn_result = RecognizerResult(
                    entity_type="SSN",
                    start=match.start(),
                    end=match.end(),
                    score=0.95,
                    analysis_explanation=None,
                    recognition_metadata=None,
                )
                results.append(ssn_result)

            # Add UK NHS detections only when explicitly referenced as NHS
            for pattern in self._nhs_context_regexes:
                for match in pattern.finditer(text):
                    if match.lastindex and match.lastindex >= 1:
                        nhs_start, nhs_end = match.start(1), match.end(1)
                        nhs_result = RecognizerResult(
                            entity_type="UK_NHS",
                            start=nhs_start,
                            end=nhs_end,
                            score=0.95,
                            analysis_explanation=None,
                            recognition_metadata=None,
                        )
                        results.append(nhs_result)

            # Add driving license detections with context
            for pattern in self._driving_license_context_regexes:
                for match in pattern.finditer(text):
                    if match.lastindex and match.lastindex >= 1:
                        dl_start, dl_end = match.start(1), match.end(1)
                        dl_result = RecognizerResult(
                            entity_type="DRIVING_LICENSE",
                            start=dl_start,
                            end=dl_end,
                            score=0.95,
                            analysis_explanation=None,
                            recognition_metadata=None,
                        )
                        results.append(dl_result)

            # Add UK-style vehicle registration detections
            for match in self._vehicle_registration_regex.finditer(text):
                vr_result = RecognizerResult(
                    entity_type="VEHICLE_REGISTRATION",
                    start=match.start(),
                    end=match.end(),
                    score=0.95,
                    analysis_explanation=None,
                    recognition_metadata=None,
                )
                results.append(vr_result)

            # Add contextual phone number detections (covers non-UK formats)
            for pattern in self._phone_context_regexes:
                for match in pattern.finditer(text):
                    if match.lastindex and match.lastindex >= 1:
                        phone_start, phone_end = match.start(1), match.end(1)
                        phone_result = RecognizerResult(
                            entity_type="PHONE_NUMBER",
                            start=phone_start,
                            end=phone_end,
                            score=0.95,
                            analysis_explanation=None,
                            recognition_metadata=None,
                        )
                        results.append(phone_result)

            # Add policy id detections with context
            for pattern in self._policy_id_context_regexes:
                for match in pattern.finditer(text):
                    if match.lastindex and match.lastindex >= 1:
                        policy_start, policy_end = match.start(1), match.end(1)
                        policy_result = RecognizerResult(
                            entity_type="POLICY_ID",
                            start=policy_start,
                            end=policy_end,
                            score=0.95,
                            analysis_explanation=None,
                            recognition_metadata=None,
                        )
                        results.append(policy_result)

            detected = sorted(
                {r.entity_type for r in results if r.score >= PII_CONFIDENCE_THRESHOLD}
            )

            # Filter results to only include high-confidence detections
            filtered_results = [r for r in results if r.score >= PII_CONFIDENCE_THRESHOLD]

            if not filtered_results:
                return SanitizationResult(
                    sanitized_text=text,
                    original_text=text,
                    detected_entities=[],
                    entity_count=0,
                )

            anonymized = anonymizer.anonymize(
                text=text,
                analyzer_results=filtered_results,
                operators=self._build_operators(),
            )

            return SanitizationResult(
                sanitized_text=anonymized.text,
                original_text=text,
                detected_entities=detected,
                entity_count=len(filtered_results),
            )

        except Exception as exc:
            logger.error("PII sanitization failed: %s", exc, exc_info=True)
            # Fall back to returning the original text rather than crashing
            return SanitizationResult(sanitized_text=text, original_text=text)
