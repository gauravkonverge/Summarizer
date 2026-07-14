"""Defensive parsing for model responses."""

import json
import re
from typing import Any


def strip_markdown_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value)
    return value.strip()


def first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_object(text: str) -> dict[str, Any]:
    cleaned = strip_markdown_fences(text)
    for candidate in (cleaned, first_json_object(cleaned)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return {}


def summary_from_response(text: str) -> str:
    payload = parse_object(text)
    for key in ("summary", "final_summary", "response", "answer", "content", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = payload.get("summary")
    if isinstance(nested, dict) and isinstance(nested.get("text"), str):
        return nested["text"].strip()
    return strip_markdown_fences(text)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip(" -\t") for part in re.split(r"\n|\|", value) if part.strip(" -\t")]
    return []


def verifier_from_response(text: str) -> dict[str, Any]:
    payload = parse_object(text)
    score = None
    for key in ("verifier_score", "score", "confidence", "rating"):
        value = payload.get(key)
        try:
            if value is not None:
                score = float(value)
                break
        except (TypeError, ValueError):
            pass
    if score is None:
        match = re.search(
            r"(?:verifier[_\s-]*score|score|confidence)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)",
            text,
            re.IGNORECASE,
        )
        score = float(match.group(1)) if match else 0.5
    reasoning = next(
        (
            payload[key].strip()
            for key in ("reasoning", "explanation", "rationale", "feedback")
            if isinstance(payload.get(key), str) and payload[key].strip()
        ),
        "Verifier response could not be fully parsed; neutral defaults were applied.",
    )
    return {
        "verifier_score": max(0.0, min(score, 1.0)),
        "reasoning": reasoning,
        "missing_points": _as_list(payload.get("missing_points")),
        "unsupported_claims": _as_list(payload.get("unsupported_claims")),
    }
