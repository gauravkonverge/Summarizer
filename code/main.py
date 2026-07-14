"""
AI-Summary API
--------------
Accepts a list of chat messages, sanitizes PII, then calls Groq to produce
a concise summary with a hybrid confidence score.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pii_sanitizer import PIISanitizer

load_dotenv()

DEFAULT_GROQ_MODEL = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.1-8b-instant")
LOG_SANITIZATION_DETAILS = os.getenv("LOG_SANITIZATION_DETAILS", "true").lower() == "true"

# Groq pricing (USD per 1M tokens) - update based on current rates
# https://console.groq.com/pricing
GROQ_INPUT_COST_PER_MTK = float(os.getenv("GROQ_INPUT_COST_PER_MTK", "0.05"))  # $0.05 per 1M input tokens
GROQ_OUTPUT_COST_PER_MTK = float(os.getenv("GROQ_OUTPUT_COST_PER_MTK", "0.15"))  # $0.15 per 1M output tokens
USD_TO_GBP_EXCHANGE_RATE = float(os.getenv("USD_TO_GBP_EXCHANGE_RATE", "0.79"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
GROQ_RETRY_BASE_DELAY_SECONDS = float(os.getenv("GROQ_RETRY_BASE_DELAY_SECONDS", "1.0"))
GROQ_RETRY_MAX_DELAY_SECONDS = float(os.getenv("GROQ_RETRY_MAX_DELAY_SECONDS", "8.0"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Summary API",
    description=(
        "Sanitizes PII from conversation messages and returns an AI-generated "
        "summary with a confidence score powered by Groq."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment variables.")
    return Groq(api_key=api_key, max_retries=0)


@lru_cache(maxsize=1)
def get_sanitizer() -> PIISanitizer:
    return PIISanitizer(language="en")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant", "system", "agent", "customer", "support"]
    content: str = Field(..., min_length=1)
    timestamp: Optional[str] = Field(None, description="ISO-8601 timestamp (optional)")
    sender: Optional[str] = Field(None, description="Display name of the sender (optional)")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class SummarizeRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    summary_style: Literal["brief", "detailed", "bullet_points"] = Field(
        default="detailed",
        description="Style of the summary",
    )
    language: str = Field(
        default="en",
        description="Language code for PII detection (ISO 639-1)",
    )


class SanitizedMessage(BaseModel):
    role: str
    original_content: str
    sanitized_content: str
    pii_detected: list[str]
    pii_count: int


class InferenceCost(BaseModel):
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")
    total_tokens: int = Field(..., description="Total tokens used")
    input_cost_usd: float = Field(..., description="Cost for input tokens in USD")
    output_cost_usd: float = Field(..., description="Cost for output tokens in USD")
    total_cost_usd: float = Field(..., description="Total cost in USD")
    input_cost_gbp_pence: float = Field(..., description="Cost for input tokens in UK pence")
    output_cost_gbp_pence: float = Field(..., description="Cost for output tokens in UK pence")
    total_cost_gbp_pence: float = Field(..., description="Total cost in UK pence")


class CallInferenceCost(BaseModel):
    input_tokens: int = Field(..., description="Number of input tokens for this call")
    output_tokens: int = Field(..., description="Number of output tokens for this call")
    total_tokens: int = Field(..., description="Total tokens for this call")
    input_cost_usd: float = Field(..., description="Input token cost for this call in USD")
    output_cost_usd: float = Field(..., description="Output token cost for this call in USD")
    total_cost_usd: float = Field(..., description="Total cost for this call in USD")
    input_cost_gbp_pence: float = Field(..., description="Input token cost for this call in UK pence")
    output_cost_gbp_pence: float = Field(..., description="Output token cost for this call in UK pence")
    total_cost_gbp_pence: float = Field(..., description="Total cost for this call in UK pence")


class InferenceCostBreakdown(BaseModel):
    groq_summary_call: CallInferenceCost
    verifier_call: CallInferenceCost
    total_both_calls: CallInferenceCost


class TimelineMetrics(BaseModel):
    """Metrics extracted from timestamped conversation for timeline analysis."""
    duration_days: int = Field(..., description="Total duration in days")
    message_count: int = Field(..., description="Total messages in conversation")
    start_date: str = Field(..., description="ISO 8601 start date")
    end_date: str = Field(..., description="ISO 8601 end date")
    support_message_count: int = Field(..., description="Count of support role messages")
    customer_message_count: int = Field(..., description="Count of customer role messages")
    unique_senders: int = Field(..., description="Count of unique senders")


class GroqCallInput(BaseModel):
    call_name: str = Field(..., description="Identifier for this Groq call (e.g. 'Pass 1 – Summary Generation')")
    model: str = Field(..., description="Model used for this call")
    temperature: float = Field(..., description="Temperature setting used")
    max_tokens: int = Field(..., description="Max tokens allowed for this call")
    system_prompt: str = Field(..., description="System prompt sent to the model")
    user_prompt: str = Field(..., description="User prompt (sanitized conversation + instructions) sent to the model")


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    summary: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    confidence_reasoning: str
    model_used: str
    sanitized_messages: list[SanitizedMessage]
    total_pii_entities_removed: int
    unique_pii_types_found: list[str]
    inference_cost: InferenceCost
    inference_cost_breakdown: InferenceCostBreakdown
    timeline_metrics: Optional[TimelineMetrics] = Field(None, description="Timeline analysis metrics (if timestamps available)")
    groq_call_inputs: list[GroqCallInput] = Field(
        default_factory=list,
        description="Input prompts sent to each Groq call in the pipeline (Pass 1 summary, Pass 2 verifier)",
    )


# ---------------------------------------------------------------------------
# Timeline helpers
# ---------------------------------------------------------------------------

def extract_timeline_metrics(messages: list[Message]) -> Optional[TimelineMetrics]:
    """
    Extract timeline metrics from timestamped messages.
    Returns None if messages lack timestamps.
    """
    timestamped_messages = [m for m in messages if m.timestamp]

    logger.info(f"Timeline extraction: total messages={len(messages)}, with timestamps={len(timestamped_messages)}")

    if len(timestamped_messages) < 2:
        return None

    try:
        timestamps = [datetime.fromisoformat(m.timestamp) for m in timestamped_messages]
        timestamps.sort()

        start_date = timestamps[0]
        end_date = timestamps[-1]
        duration_days = (end_date - start_date).days

        support_count = sum(1 for m in messages if m.role == "support")
        customer_count = sum(1 for m in messages if m.role == "customer")
        unique_senders = len(set(m.sender for m in messages if m.sender))

        logger.info(
            f"Timeline metrics computed: duration={duration_days}d, support={support_count}, "
            f"customer={customer_count}, unique_senders={unique_senders}"
        )

        return TimelineMetrics(
            duration_days=duration_days,
            message_count=len(messages),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            support_message_count=support_count,
            customer_message_count=customer_count,
            unique_senders=unique_senders,
        )
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse timestamps for timeline metrics: {e}")
        return None


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------

STYLE_INSTRUCTIONS = {
    "brief": "Write a brief 2-3 sentence summary.",
    "detailed": "Write a detailed paragraph summary covering all key points.",
    "bullet_points": "Write a summary as a structured bullet-point list of key topics.",
}

SYSTEM_PROMPT = """\
You are an expert communication analyst. You will be given a conversation with \
PII already sanitized (replaced with labels like [PERSON], [EMAIL], etc.).

Your task:
1. Summarize the conversation according to the requested style.
2. Preserve only facts supported by the conversation.
3. Do not invent names, dates, causes, or outcomes that are not present.

Respond ONLY in the following JSON structure (no markdown fences):
{
  "summary": "<your summary here>"
}
"""

VERIFIER_SYSTEM_PROMPT = """\
You are a strict QA verifier for conversation summaries.

You will receive:
1. A sanitized conversation
2. A generated summary

Evaluate whether the summary is faithful to the conversation.
Penalize the score when the summary:
- misses important points
- introduces unsupported claims
- overstates certainty
- becomes vague because redaction removed critical context

Respond ONLY in the following JSON structure (no markdown fences):
{
  "verifier_score": <number between 0 and 1>,
  "reasoning": "<brief explanation>",
  "missing_points": ["<optional missing point>"],
  "unsupported_claims": ["<optional unsupported claim>"]
}
"""


def clamp_score(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def strip_markdown_fences(raw_content: str) -> str:
    if raw_content.startswith("```"):
        raw_content = re.sub(r"^```[a-z]*\n?", "", raw_content)
        raw_content = re.sub(r"\n?```$", "", raw_content)
    return raw_content.strip()


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Return the first balanced JSON object found in text, or None.
    Handles quoted strings so braces inside strings are ignored.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
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
                return text[start : idx + 1]

    return None


def parse_json_response(raw_content: str, fallback_payload: dict, warning_message: str) -> dict:
    cleaned_content = strip_markdown_fences(raw_content)
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        json_candidate = extract_first_json_object(cleaned_content)
        if json_candidate:
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

        logger.warning(warning_message)
        return fallback_payload


def try_parse_json_response(raw_content: str) -> tuple[dict, bool]:
    """
    Attempt to parse JSON without logging warnings.
    Returns (payload, parsed_ok).
    """
    cleaned_content = strip_markdown_fences(raw_content)
    try:
        return json.loads(cleaned_content), True
    except json.JSONDecodeError:
        json_candidate = extract_first_json_object(cleaned_content)
        if json_candidate:
            try:
                return json.loads(json_candidate), True
            except json.JSONDecodeError:
                pass
    return {}, False


def extract_message_content_text(response) -> str:
    """
    Normalize provider response message content into plain text.
    Supports both plain string content and list-of-part content blocks.
    """
    message = response.choices[0].message

    def collect_text_candidates(obj, candidates: list[str]) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            text = obj.strip()
            if text:
                candidates.append(text)
            return
        if isinstance(obj, dict):
            preferred_order = (
                "summary",
                "text",
                "content",
                "output_text",
                "response",
                "answer",
                "final",
                "reasoning",
            )
            for key in preferred_order:
                if key in obj:
                    collect_text_candidates(obj.get(key), candidates)
            for key, value in obj.items():
                if key not in preferred_order:
                    collect_text_candidates(value, candidates)
            return
        if isinstance(obj, list):
            for item in obj:
                collect_text_candidates(item, candidates)
            return

        if hasattr(obj, "model_dump"):
            try:
                collect_text_candidates(obj.model_dump(), candidates)
                return
            except Exception:
                pass

        if hasattr(obj, "__dict__"):
            try:
                collect_text_candidates(vars(obj), candidates)
                return
            except Exception:
                pass

        text = str(obj).strip()
        if text and text not in {"[]", "{}", "None"}:
            candidates.append(text)
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text") or ""
                if text_value:
                    text_parts.append(str(text_value))
            else:
                text_value = getattr(part, "text", "")
                if text_value:
                    text_parts.append(str(text_value))
        return "\n".join(text_parts).strip()

    text = str(content or "").strip()
    if text:
        return text

    # Some models may place usable text in reasoning fields instead of content.
    for attr in ("reasoning", "reasoning_content"):
        reasoning_value = getattr(message, attr, "")
        if isinstance(reasoning_value, str) and reasoning_value.strip():
            return reasoning_value.strip()

    candidates: list[str] = []
    collect_text_candidates(message, candidates)
    if candidates:
        # Prefer the longest candidate to avoid returning tiny metadata fragments.
        return max(candidates, key=len)

    return ""


def extract_summary_text(parsed_payload: dict, raw_content: str) -> str:
    """
    Extract summary text from multiple potential payload shapes.
    Falls back to raw content to avoid empty summaries.
    """
    preferred_keys = (
        "summary",
        "final_summary",
        "response",
        "answer",
        "content",
        "text",
    )

    for key in preferred_keys:
        value = parsed_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(parsed_payload.get("summary"), dict):
        nested_summary = parsed_payload["summary"].get("text")
        if isinstance(nested_summary, str) and nested_summary.strip():
            return nested_summary.strip()

    return strip_markdown_fences(raw_content)


def _list_from_value(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        chunks = [chunk.strip(" -\t") for chunk in re.split(r"\n|\|", value)]
        return [chunk for chunk in chunks if chunk]
    return []


def _find_score_in_text(text: str) -> Optional[float]:
    if not text:
        return None

    labeled_match = re.search(
        r"(?:verifier[_\s-]*score|score|confidence)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if labeled_match:
        return float(labeled_match.group(1))

    bare_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if bare_match:
        return float(bare_match.group(1))

    return None


def normalize_verifier_result(parsed_payload: dict, raw_content: str) -> dict:
    score: Optional[float] = None
    reasoning = ""
    missing_points: list[str] = []
    unsupported_claims: list[str] = []

    for key in ("verifier_score", "score", "confidence", "rating"):
        value = parsed_payload.get(key)
        if isinstance(value, (int, float)):
            score = float(value)
            break
        if isinstance(value, str):
            parsed_score = _find_score_in_text(value)
            if parsed_score is not None:
                score = parsed_score
                break

    if score is None:
        score = _find_score_in_text(raw_content)

    for key in ("reasoning", "explanation", "rationale", "feedback"):
        value = parsed_payload.get(key)
        if isinstance(value, str) and value.strip():
            reasoning = value.strip()
            break

    if not reasoning:
        reasoning_match = re.search(r"(?:reasoning|explanation|rationale)\s*[:=]\s*(.+)", raw_content, flags=re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

    missing_points = _list_from_value(parsed_payload.get("missing_points"))
    unsupported_claims = _list_from_value(parsed_payload.get("unsupported_claims"))

    if not missing_points:
        missing_match = re.search(r"missing[_\s-]*points\s*[:=]\s*(.+)", raw_content, flags=re.IGNORECASE)
        if missing_match:
            missing_points = _list_from_value(missing_match.group(1))

    if not unsupported_claims:
        unsupported_match = re.search(r"unsupported[_\s-]*claims\s*[:=]\s*(.+)", raw_content, flags=re.IGNORECASE)
        if unsupported_match:
            unsupported_claims = _list_from_value(unsupported_match.group(1))

    result = {
        "verifier_score": 0.5 if score is None else clamp_score(score),
        "reasoning": reasoning or "Verifier response parsed from fallback format.",
        "missing_points": missing_points,
        "unsupported_claims": unsupported_claims,
    }

    if score is None:
        result["reasoning"] = "Verifier response could not be parsed; using neutral verifier score."

    return result


def extract_usage_info(response) -> dict:
    return {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }


def merge_usage_info(*usage_dicts: dict) -> dict:
    return {
        "input_tokens": sum(usage.get("input_tokens", 0) for usage in usage_dicts),
        "output_tokens": sum(usage.get("output_tokens", 0) for usage in usage_dicts),
        "total_tokens": sum(usage.get("total_tokens", 0) for usage in usage_dicts),
    }


def is_json_mode_unsupported_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code != 400:
        return False
    message = str(exc).lower()
    return "json_validate_failed" in message or "failed to validate json" in message


def create_groq_completion_with_json_fallback(client: Groq, **kwargs):
    """Try with response_format json_object; if the model rejects it, retry without."""
    try:
        return create_groq_completion_with_retry(client, **kwargs)
    except Exception as exc:
        if is_json_mode_unsupported_error(exc) and "response_format" in kwargs:
            logger.warning(
                "Model does not support response_format=json_object; retrying without it."
            )
            kwargs_without_fmt = {k: v for k, v in kwargs.items() if k != "response_format"}
            return create_groq_completion_with_retry(client, **kwargs_without_fmt)
        raise


def is_rate_limited_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    message = str(exc).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message


def create_groq_completion_with_retry(client: Groq, **kwargs):
    max_attempts = max(1, GROQ_MAX_RETRIES)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            if not is_rate_limited_error(exc) or attempt == max_attempts:
                raise

            delay_seconds = min(
                GROQ_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                GROQ_RETRY_MAX_DELAY_SECONDS,
            )
            logger.warning(
                "Groq rate limit hit (attempt %d/%d). Retrying in %.1f second(s).",
                attempt,
                max_attempts,
                delay_seconds,
            )
            time.sleep(delay_seconds)

    if last_error:
        raise last_error
    raise RuntimeError("Groq completion failed without a captured error.")


def usd_to_gbp_pence(usd_amount: float) -> float:
    return usd_amount * USD_TO_GBP_EXCHANGE_RATE * 100


def build_call_inference_cost(usage_info: dict) -> CallInferenceCost:
    input_tokens = usage_info.get("input_tokens", 0)
    output_tokens = usage_info.get("output_tokens", 0)
    total_tokens = usage_info.get("total_tokens", 0)

    input_cost_usd = (input_tokens / 1_000_000) * GROQ_INPUT_COST_PER_MTK
    output_cost_usd = (output_tokens / 1_000_000) * GROQ_OUTPUT_COST_PER_MTK
    total_cost_usd = input_cost_usd + output_cost_usd

    input_cost_gbp_pence = usd_to_gbp_pence(input_cost_usd)
    output_cost_gbp_pence = usd_to_gbp_pence(output_cost_usd)
    total_cost_gbp_pence = usd_to_gbp_pence(total_cost_usd)

    return CallInferenceCost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost_usd=round(input_cost_usd, 6),
        output_cost_usd=round(output_cost_usd, 6),
        total_cost_usd=round(total_cost_usd, 6),
        input_cost_gbp_pence=round(input_cost_gbp_pence, 4),
        output_cost_gbp_pence=round(output_cost_gbp_pence, 4),
        total_cost_gbp_pence=round(total_cost_gbp_pence, 4),
    )

def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def compute_rule_based_confidence(messages: list[SanitizedMessage]) -> tuple[float, str]:
    score = 0.92
    reasons: list[str] = []

    original_text = "\n".join(message.original_content for message in messages)
    sanitized_text = "\n".join(message.sanitized_content for message in messages)
    total_words = max(1, count_words(original_text))
    avg_words_per_message = sum(count_words(message.sanitized_content) for message in messages) / max(1, len(messages))
    meaningful_messages = sum(1 for message in messages if count_words(message.sanitized_content) >= 4)
    placeholder_count = len(re.findall(r"\[[A-Z_]+\]", sanitized_text))
    redaction_density = placeholder_count / total_words
    pii_type_count = len({pii_type for message in messages for pii_type in message.pii_detected})

    if len(messages) < 2:
        score -= 0.25
        reasons.append("very short conversation")
    elif len(messages) < 4:
        score -= 0.10
        reasons.append("limited conversation length")

    if meaningful_messages < max(1, len(messages) // 2):
        score -= 0.12
        reasons.append("few substantive messages")

    if avg_words_per_message < 5:
        score -= 0.08
        reasons.append("messages are brief")

    if redaction_density > 0.12:
        score -= 0.18
        reasons.append("heavy PII redaction removed context")
    elif redaction_density > 0.06:
        score -= 0.10
        reasons.append("moderate PII redaction removed some context")

    if pii_type_count >= 6:
        score -= 0.05
        reasons.append("many PII categories were sanitized")

    if not reasons:
        reasons.append("conversation is sufficiently detailed with limited redaction impact")

    return round(clamp_score(score), 2), "; ".join(reasons)


def build_conversation_text(messages: list[SanitizedMessage]) -> str:
    lines = []
    for msg in messages:
        role_label = msg.role.upper()
        lines.append(f"[{role_label}]: {msg.sanitized_content}")
    return "\n".join(lines)


def call_groq_for_summary(
    conversation_text: str,
    style: str,
    model: str,
) -> tuple[dict, dict, str, str]:
    """
    Call Groq for summary and return parsed result, usage info, system prompt, and user prompt.
    Returns (parsed_result, usage_dict, system_prompt, user_prompt)
    """
    client = get_groq_client()
    user_prompt = (
        f"Conversation:\n{conversation_text}\n\n"
        f"Summary style: {STYLE_INSTRUCTIONS[style]}\n\n"
        "Respond with the JSON structure defined in your instructions."
    )

    response = create_groq_completion_with_json_fallback(
        client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw_content = extract_message_content_text(response)
    parsed = parse_json_response(
        raw_content=raw_content,
        fallback_payload={"summary": strip_markdown_fences(raw_content)},
        warning_message="Groq summary response was not valid JSON; using raw text as summary.",
    )

    normalized_summary = extract_summary_text(parsed, raw_content)
    usage_dict = extract_usage_info(response)

    if not normalized_summary.strip():
        logger.warning("Primary summary response was empty; retrying with plain-text fallback prompt.")
        fallback_prompt = (
            f"Conversation:\n{conversation_text}\n\n"
            f"Summary style: {STYLE_INSTRUCTIONS[style]}\n\n"
            "Return ONLY the summary as plain text. Do not return JSON or markdown."
        )

        fallback_response = create_groq_completion_with_retry(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert communication analyst. Return a factual summary only.",
                },
                {"role": "user", "content": fallback_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        fallback_raw = extract_message_content_text(fallback_response)
        normalized_summary = strip_markdown_fences(fallback_raw)
        usage_dict = merge_usage_info(usage_dict, extract_usage_info(fallback_response))

        if not normalized_summary.strip():
            normalized_summary = "Summary unavailable: model returned empty content."

    parsed = {"summary": normalized_summary}

    return parsed, usage_dict, SYSTEM_PROMPT, user_prompt


def call_groq_for_verification(
    conversation_text: str,
    summary_text: str,
    model: str,
) -> tuple[dict, dict, str, str]:
    """
    Returns (verifier_result, usage_dict, system_prompt, user_prompt)
    """
    client = get_groq_client()
    user_prompt = (
        f"Conversation:\n{conversation_text}\n\n"
        f"Summary to verify:\n{summary_text}\n\n"
        "Verify the summary against the conversation and respond with the JSON structure defined in your instructions."
    )

    response = create_groq_completion_with_json_fallback(
        client,
        model=model,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw_content = extract_message_content_text(response)
    parsed, _ = try_parse_json_response(raw_content)
    normalized = normalize_verifier_result(parsed, raw_content)
    usage_dict = extract_usage_info(response)

    return normalized, usage_dict, VERIFIER_SYSTEM_PROMPT, user_prompt


def combine_confidence_scores(
    rule_score: float,
    rule_reasoning: str,
    verifier_result: dict,
) -> tuple[float, str]:
    verifier_score = clamp_score(float(verifier_result.get("verifier_score", 0.5)))
    missing_points = verifier_result.get("missing_points", []) or []
    unsupported_claims = verifier_result.get("unsupported_claims", []) or []

    final_score = round(clamp_score((rule_score * 0.4) + (verifier_score * 0.6)), 2)

    reasoning_parts = [
        f"rule score={rule_score:.2f} ({rule_reasoning})",
        f"verifier score={verifier_score:.2f} ({verifier_result.get('reasoning', 'no verifier reasoning provided')})",
    ]

    if missing_points:
        reasoning_parts.append(f"missing points: {', '.join(missing_points[:2])}")

    if unsupported_claims:
        reasoning_parts.append(f"unsupported claims: {', '.join(unsupported_claims[:2])}")

    return final_score, "; ".join(reasoning_parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"])
def health_check():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/docs/swagger/index", include_in_schema=False)
def swagger_compat_redirect() -> RedirectResponse:
    """Compatibility route for Swagger-style URL patterns."""
    return RedirectResponse(url="/docs")


@app.post(
    "/api/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Summarization"],
    summary="Sanitize PII and summarize conversation messages",
)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    """
    **Workflow:**
    1. Each message is scanned for PII using Microsoft Presidio.
    2. Detected PII is replaced with labeled placeholders (e.g. `[EMAIL]`).
    3. The sanitized conversation is sent to Groq for summarization.
     4. The response includes the summary, a hybrid 0-1 confidence score, and
         details of which PII types were removed from each message.
    """
    sanitizer = get_sanitizer()

    # --- Step 1: Sanitize all messages ---
    sanitized_messages: list[SanitizedMessage] = []
    all_pii_types: set[str] = set()
    total_pii = 0

    for idx, msg in enumerate(request.messages, start=1):
        result = sanitizer.sanitize(msg.content)
        sanitized_messages.append(
            SanitizedMessage(
                role=msg.role,
                original_content=msg.content,
                sanitized_content=result.sanitized_text,
                pii_detected=result.detected_entities,
                pii_count=result.entity_count,
            )
        )
        all_pii_types.update(result.detected_entities)
        total_pii += result.entity_count

        if LOG_SANITIZATION_DETAILS:
            logger.info(
                "Sanitization detail | message=%d role=%s before=%r after=%r pii_types=%s pii_count=%d",
                idx,
                msg.role,
                msg.content,
                result.sanitized_text,
                result.detected_entities,
                result.entity_count,
            )

    logger.info(
        "Sanitized %d message(s). Total PII entities removed: %d. Types: %s",
        len(sanitized_messages),
        total_pii,
        sorted(all_pii_types),
    )

    # --- Step 2: Build conversation text from sanitized messages ---
    conversation_text = build_conversation_text(sanitized_messages)

    # --- Step 3: Resolve model from configuration ---
    selected_model = DEFAULT_GROQ_MODEL

    # --- Step 4: Call Groq ---
    try:
        groq_result, summary_usage, summary_sys_prompt, summary_user_prompt = call_groq_for_summary(
            conversation_text=conversation_text,
            style=request.summary_style,
            model=selected_model,
        )

        verifier_result, verifier_usage, verifier_sys_prompt, verifier_user_prompt = call_groq_for_verification(
            conversation_text=conversation_text,
            summary_text=groq_result.get("summary", ""),
            model=selected_model,
        )
    except Exception as exc:
        logger.error("Groq API call failed: %s", exc, exc_info=True)
        if is_rate_limited_error(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "Groq rate limit exceeded. Please retry shortly or reduce request frequency. "
                    "You can also increase retry settings via GROQ_MAX_RETRIES and retry delay env vars."
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get summary from Groq: {exc}",
        )

    summary_call_cost = build_call_inference_cost(summary_usage)
    verifier_call_cost = build_call_inference_cost(verifier_usage)
    usage_info = merge_usage_info(summary_usage, verifier_usage)
    total_both_calls_cost = build_call_inference_cost(usage_info)

    # --- Step 5: Calculate inference cost ---
    input_tokens = usage_info.get("input_tokens", 0)
    output_tokens = usage_info.get("output_tokens", 0)
    total_tokens = usage_info.get("total_tokens", 0)

    input_cost = (input_tokens / 1_000_000) * GROQ_INPUT_COST_PER_MTK
    output_cost = (output_tokens / 1_000_000) * GROQ_OUTPUT_COST_PER_MTK
    total_cost = input_cost + output_cost

    inference_cost = InferenceCost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(total_cost, 6),
        input_cost_gbp_pence=round(usd_to_gbp_pence(input_cost), 4),
        output_cost_gbp_pence=round(usd_to_gbp_pence(output_cost), 4),
        total_cost_gbp_pence=round(usd_to_gbp_pence(total_cost), 4),
    )

    logger.info(
        "Inference cost total: input=%d tokens ($%.6f), output=%d tokens ($%.6f), total=$%.6f",
        input_tokens, input_cost, output_tokens, output_cost, total_cost
    )
    logger.info(
        "Groq summary call: input=%d, output=%d, total=%d, total_usd=%.6f, total_gbp_pence=%.4f",
        summary_call_cost.input_tokens,
        summary_call_cost.output_tokens,
        summary_call_cost.total_tokens,
        summary_call_cost.total_cost_usd,
        summary_call_cost.total_cost_gbp_pence,
    )
    logger.info(
        "Groq verifier call: input=%d, output=%d, total=%d, total_usd=%.6f, total_gbp_pence=%.4f",
        verifier_call_cost.input_tokens,
        verifier_call_cost.output_tokens,
        verifier_call_cost.total_tokens,
        verifier_call_cost.total_cost_usd,
        verifier_call_cost.total_cost_gbp_pence,
    )

    # --- Step 6: Compute hybrid confidence ---
    rule_score, rule_reasoning = compute_rule_based_confidence(sanitized_messages)
    confidence, confidence_reasoning = combine_confidence_scores(
        rule_score=rule_score,
        rule_reasoning=rule_reasoning,
        verifier_result=verifier_result,
    )

    # --- Step 7: Extract timeline metrics ---
    timeline_metrics = extract_timeline_metrics(request.messages)

    return SummarizeResponse(
        summary=groq_result.get("summary", ""),
        confidence=confidence,
        confidence_reasoning=confidence_reasoning,
        model_used=selected_model,
        sanitized_messages=sanitized_messages,
        total_pii_entities_removed=total_pii,
        unique_pii_types_found=sorted(all_pii_types),
        inference_cost=inference_cost,
        inference_cost_breakdown=InferenceCostBreakdown(
            groq_summary_call=summary_call_cost,
            verifier_call=verifier_call_cost,
            total_both_calls=total_both_calls_cost,
        ),
        timeline_metrics=timeline_metrics,
        groq_call_inputs=[
            GroqCallInput(
                call_name="Pass 1 – Summary Generation",
                model=selected_model,
                temperature=0.3,
                max_tokens=1024,
                system_prompt=summary_sys_prompt,
                user_prompt=summary_user_prompt,
            ),
            GroqCallInput(
                call_name="Pass 2 – Summary Verification",
                model=selected_model,
                temperature=0.0,
                max_tokens=512,
                system_prompt=verifier_sys_prompt,
                user_prompt=verifier_user_prompt,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
