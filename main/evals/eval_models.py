"""Evaluate candidate Bedrock models on speed and summary quality.

For each sample: real Guardrail sanitization + garbled detection (as in the
pipeline), then each candidate runs the production summary and verifier
prompts. An independent judge model grades each summary against planted key
facts and traps.
"""

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import logging
import statistics
import sys
import time

samples_path, out_path = sys.argv[1], sys.argv[2]
logging.basicConfig(level=logging.ERROR)

from app.core.aws import bedrock_runtime_client  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.providers.base import LLMResult, TokenUsage  # noqa: E402
from app.models.schemas import SanitizedMessage, SummarizeRequest  # noqa: E402
from app.providers.bedrock import BedrockProvider  # noqa: E402
from app.services.guardrail import BedrockGuardrailSanitizer  # noqa: E402
from app.services.parsing import parse_object, summary_from_response, verifier_from_response  # noqa: E402
from app.services.pipeline import SUMMARY_MAX_TOKENS, VERIFIER_MAX_TOKENS, SummarizationPipeline  # noqa: E402
from app.services.prompts import (  # noqa: E402
    SUMMARY_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    summary_user_prompt,
    verifier_user_prompt,
)
from app.services.text_quality import assess_garbled  # noqa: E402

# (label, Bedrock model id, additionalModelRequestFields or None)
CANDIDATES = [
    ("haiku-4.5", "us.anthropic.claude-haiku-4-5-20251001-v1:0", None),
    ("haiku-5.5 effort=low", "us.anthropic.claude-haiku-5-5", {"output_config": {"effort": "low"}}),
    ("gpt-5.6-terra", "us.openai.gpt-5.6-terra", None),
    ("gpt-6.1-terra", "us.openai.gpt-6.1-terra", None),
    ("gpt-6-luna", "us.openai.gpt-6-luna", None),
    ("gpt-oss-20b", "openai.gpt-oss-20b-1:0", None),
]
JUDGE = "us.anthropic.claude-opus-4-5-20251101-v1:0"

JUDGE_SYSTEM = """You grade conversation summaries for a fleet-services summarization API.
You are given the conversation (PII already masked), the requested style and language,
a list of KEY FACTS the summary should cover, TRAPS it must not fall into, and the summary.
Masked placeholders such as [PERSON], [EMAIL] or [CREDIT_CARD] are NOT PII leaks.
Mentioning an earlier, superseded value as history is fine if the final value is clear.
Grade strictly. Respond only as JSON:
{
  "facts_covered": [true/false per key fact, same order],
  "trap_violations": ["<trap text that was violated>", ...],
  "hallucinations": ["<claim not supported by the conversation>", ...],
  "style_ok": true/false,      // matches requested style (brief = 2-3 sentences, bullet_points = bullet list, detailed = paragraph)
  "language_ok": true/false,   // written in the requested language
  "score": <integer 1-5 overall quality>,
  "notes": "<one sentence>"
}"""

base = dataclasses.replace(Settings(), llm_max_attempts=3)
samples = json.load(open(samples_path, encoding="utf-8"))
sanitizer = BedrockGuardrailSanitizer(base)


def prepare(sample: dict) -> dict:
    request = SummarizeRequest.model_validate(sample["request"])
    messages = []
    pii = 0
    for message in request.messages:
        result = sanitizer.sanitize(message.content)
        garbled, reason = assess_garbled(result.sanitized_text)
        pii += result.entity_count
        messages.append(
            SanitizedMessage(
                role=message.role, original_content="", sanitized_content=result.sanitized_text,
                pii_detected=result.detected_entities, pii_count=result.entity_count,
                is_garbled=garbled, garbled_reason=reason,
            )
        )
    conversation = SummarizationPipeline._conversation(messages)
    return {
        **sample,
        "conversation": conversation,
        "pii_masked": pii,
        "garbled": sum(m.is_garbled for m in messages),
        "summary_prompt": summary_user_prompt(conversation, request.summary_style, request.language),
    }


client = bedrock_runtime_client(base, sdk_retries=True)


def converse(model_id: str, extra: dict | None, system: str, user: str, max_tokens: int) -> LLMResult:
    kwargs = dict(
        modelId=model_id,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    if extra:
        kwargs["additionalModelRequestFields"] = extra
    response = client.converse(**kwargs)
    blocks = response["output"]["message"]["content"]
    usage = response["usage"]
    return LLMResult(
        text="\n".join(b["text"] for b in blocks if "text" in b).strip(),
        usage=TokenUsage(usage["inputTokens"], usage["outputTokens"], usage["totalTokens"]),
    )


def run_candidate(candidate: tuple, sample: dict) -> dict:
    label, model_id, extra = candidate
    row = {"model": label, "model_id": model_id, "sample": sample["id"], "error": None}
    try:
        started = time.perf_counter()
        res = converse(model_id, extra, SUMMARY_SYSTEM_PROMPT, sample["summary_prompt"], SUMMARY_MAX_TOKENS)
        row["summary_ms"] = (time.perf_counter() - started) * 1000
        summary = summary_from_response(res.text)
        row["json_ok"] = bool(summary.strip())
        row["summary"] = summary or res.text
        row["out_tokens"] = res.usage.output_tokens
        row["in_tokens"] = res.usage.input_tokens
        started = time.perf_counter()
        ver = converse(
            model_id, extra, VERIFIER_SYSTEM_PROMPT,
            verifier_user_prompt(sample["conversation"], row["summary"]), VERIFIER_MAX_TOKENS,
        )
        row["verifier_ms"] = (time.perf_counter() - started) * 1000
        parsed = verifier_from_response(ver.text)
        row["verifier_ok"] = "could not be fully parsed" not in parsed["reasoning"]
        row["verifier_score"] = parsed["verifier_score"]
    except Exception as exc:  # noqa: BLE001
        cause = exc.__cause__ or exc
        row["error"] = f"{type(cause).__name__}: {str(cause)[:200]}"
    return row


def judge(row: dict, sample: dict) -> dict:
    if row["error"]:
        return row
    provider = BedrockProvider(dataclasses.replace(base, bedrock_model_id=JUDGE))
    req = sample["request"]
    prompt = (
        f"<conversation>\n{sample['conversation']}\n</conversation>\n\n"
        f"Requested style: {req['summary_style']}\nRequested language: {req['language']}\n\n"
        "KEY FACTS:\n" + "\n".join(f"{i + 1}. {f}" for i, f in enumerate(sample["key_facts"])) + "\n\n"
        "TRAPS:\n" + "\n".join(f"- {t}" for t in sample["traps"]) + "\n\n"
        f"<summary>\n{row['summary']}\n</summary>"
    )
    res = provider.complete(system_prompt=JUDGE_SYSTEM, user_prompt=prompt, max_tokens=1024, temperature=0)
    verdict = parse_object(res.text)
    covered = verdict.get("facts_covered") or []
    row["judge"] = verdict
    row["fact_coverage"] = sum(bool(c) for c in covered) / len(sample["key_facts"])
    row["trap_violations"] = len(verdict.get("trap_violations") or [])
    row["hallucinations"] = len(verdict.get("hallucinations") or [])
    row["judge_score"] = verdict.get("score")
    return row


with ThreadPoolExecutor(max_workers=6) as pool:
    prepared = list(pool.map(prepare, samples))
by_id = {s["id"]: s for s in prepared}
for s in prepared:
    print(f"{s['id']}: pii_masked={s['pii_masked']} garbled={s['garbled']}")

# Candidates run in parallel with each other, but each model's calls run one at a time.
def run_model(candidate: tuple) -> list[dict]:
    return [run_candidate(candidate, s) for s in prepared]

with ThreadPoolExecutor(max_workers=len(CANDIDATES)) as pool:
    rows = [row for model_rows in pool.map(run_model, CANDIDATES) for row in model_rows]
with ThreadPoolExecutor(max_workers=6) as pool:
    rows = list(pool.map(lambda r: judge(r, by_id[r["sample"]]), rows))

json.dump({"samples": prepared, "rows": rows}, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print(f"\n{'model':24} {'sum p50':>8} {'ver p50':>8} {'total':>7} {'facts':>6} {'traps':>5} {'halluc':>6} {'score':>5} errs")
for model_id, _id, _extra in CANDIDATES:
    ok = [r for r in rows if r["model"] == model_id and not r["error"]]
    errs = sum(1 for r in rows if r["model"] == model_id and r["error"])
    if not ok:
        print(f"{model_id:24} all failed")
        continue
    s50 = statistics.median(r["summary_ms"] for r in ok)
    v50 = statistics.median(r["verifier_ms"] for r in ok)
    print(
        f"{model_id:24} {s50:>6.0f}ms {v50:>6.0f}ms {s50 + v50:>5.0f}ms "
        f"{statistics.mean(r['fact_coverage'] for r in ok):>6.0%} "
        f"{sum(r['trap_violations'] for r in ok):>5} {sum(r['hallucinations'] for r in ok):>6} "
        f"{statistics.mean(r['judge_score'] or 0 for r in ok):>5.1f} {errs}"
    )
