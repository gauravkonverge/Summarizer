"""Compare candidate Bedrock models on the production summary/verifier prompts."""

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import logging
import statistics
import sys
import time

sample_path, out_path = sys.argv[1], sys.argv[2]
logging.basicConfig(level=logging.ERROR)

from app.core.config import Settings  # noqa: E402
from app.models.schemas import SanitizedMessage, SummarizeRequest  # noqa: E402
from app.providers.bedrock import BedrockProvider  # noqa: E402
from app.services.parsing import summary_from_response, verifier_from_response  # noqa: E402
from app.services.pipeline import (  # noqa: E402
    SUMMARY_MAX_TOKENS,
    VERIFIER_MAX_TOKENS,
    SummarizationPipeline,
)
from app.services.prompts import (  # noqa: E402
    SUMMARY_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    summary_user_prompt,
    verifier_user_prompt,
)

MODELS = [
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",  # current baseline
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-5-5",
    "us.amazon.nova-2-lite-v1:0",
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-micro-v1:0",
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    "us.meta.llama3-3-70b-instruct-v1:0",
    "openai.gpt-oss-120b-1:0",
    "qwen.qwen3-next-80b-a3b",
    "mistral.ministral-3-14b-instruct",
    "google.gemma-3-27b-it",
]
RUNS = 2

settings = dataclasses.replace(Settings(), llm_max_attempts=1)
request = SummarizeRequest.model_validate(json.load(open(sample_path, encoding="utf-8")))
# The guardrail found no PII in this sample, so content is used unchanged.
conversation = SummarizationPipeline._conversation(
    [
        SanitizedMessage(
            role=m.role, original_content="", sanitized_content=m.content,
            pii_detected=[], pii_count=0, is_garbled=False, garbled_reason=None,
        )
        for m in request.messages
    ]
)
summary_prompt = summary_user_prompt(conversation, request.summary_style, request.language)


def bench(model_id: str) -> dict:
    provider = BedrockProvider(dataclasses.replace(settings, bedrock_model_id=model_id))
    result: dict = {"model": model_id, "summary_ms": [], "error": None}
    try:
        provider.warm_up()
        for _ in range(RUNS):
            started = time.perf_counter()
            res = provider.complete(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=summary_prompt,
                max_tokens=SUMMARY_MAX_TOKENS,
            )
            result["summary_ms"].append((time.perf_counter() - started) * 1000)
        summary = summary_from_response(res.text)
        result.update(
            json_ok=bool(summary.strip()),
            out_tokens=res.usage.output_tokens,
            summary=summary or res.text,
        )
        started = time.perf_counter()
        ver = provider.complete(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=verifier_user_prompt(conversation, summary or res.text),
            max_tokens=VERIFIER_MAX_TOKENS,
        )
        result["verifier_ms"] = (time.perf_counter() - started) * 1000
        parsed = verifier_from_response(ver.text)
        result["verifier_parsed"] = "could not be fully parsed" not in parsed["reasoning"]
        result["verifier_score"] = parsed["verifier_score"]
    except Exception as exc:  # noqa: BLE001
        cause = exc.__cause__ or exc
        result["error"] = f"{type(cause).__name__}: {str(cause)[:160]}"
    return result


with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(bench, MODELS))

json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2)
print(f"{'model':48} {'summary med':>11} {'verifier':>9} {'out tok':>7} json ver")
for r in sorted(results, key=lambda r: statistics.median(r["summary_ms"]) if r["summary_ms"] and not r["error"] else 1e9):
    if r["error"]:
        print(f"{r['model']:48} ERROR {r['error']}")
        continue
    print(
        f"{r['model']:48} {statistics.median(r['summary_ms']):>9.0f}ms "
        f"{r['verifier_ms']:>7.0f}ms {r['out_tokens']:>7} {str(r['json_ok']):>4} {str(r['verifier_parsed']):>4}"
    )
