"""Rule-based and verifier-assisted confidence scoring."""

import re
from typing import Any

from app.models.schemas import SanitizedMessage


def _clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def rule_based_confidence(messages: list[SanitizedMessage]) -> tuple[float, str]:
    score = 0.92
    reasons: list[str] = []
    original = "\n".join(message.original_content for message in messages)
    sanitized = "\n".join(message.sanitized_content for message in messages)
    total_words = max(1, _word_count(original))
    average_words = sum(_word_count(message.sanitized_content) for message in messages) / len(messages)
    meaningful = sum(_word_count(message.sanitized_content) >= 4 for message in messages)
    redaction_density = len(re.findall(r"\[[A-Z_]+\]", sanitized)) / total_words
    pii_types = {entity for message in messages for entity in message.pii_detected}

    if len(messages) < 2:
        score -= 0.25
        reasons.append("very short conversation")
    elif len(messages) < 4:
        score -= 0.10
        reasons.append("limited conversation length")
    if meaningful < max(1, len(messages) // 2):
        score -= 0.12
        reasons.append("few substantive messages")
    if average_words < 5:
        score -= 0.08
        reasons.append("messages are brief")
    if redaction_density > 0.12:
        score -= 0.18
        reasons.append("heavy PII redaction removed context")
    elif redaction_density > 0.06:
        score -= 0.10
        reasons.append("moderate PII redaction removed some context")
    if len(pii_types) >= 6:
        score -= 0.05
        reasons.append("many PII categories were sanitized")
    if not reasons:
        reasons.append("conversation is sufficiently detailed with limited redaction impact")
    return round(_clamp(score), 2), "; ".join(reasons)


def combined_confidence(
    rule_score: float, rule_reasoning: str, verifier: dict[str, Any]
) -> tuple[float, str]:
    verifier_score = _clamp(float(verifier.get("verifier_score", 0.5)))
    final = round(_clamp(rule_score * 0.4 + verifier_score * 0.6), 2)
    parts = [
        f"rule score={rule_score:.2f} ({rule_reasoning})",
        f"verifier score={verifier_score:.2f} ({verifier.get('reasoning', 'no reasoning provided')})",
    ]
    missing = verifier.get("missing_points") or []
    unsupported = verifier.get("unsupported_claims") or []
    if missing:
        parts.append(f"missing points: {', '.join(str(item) for item in missing[:2])}")
    if unsupported:
        parts.append(f"unsupported claims: {', '.join(str(item) for item in unsupported[:2])}")
    return final, "; ".join(parts)
