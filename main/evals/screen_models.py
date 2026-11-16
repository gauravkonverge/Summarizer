"""Quick latency screen: one summary call per model on one conversation (live Bedrock).

Usage: PYTHONPATH=. .venv/Scripts/python.exe evals/screen_models.py evals/rac_sample.json
"""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import sys
import time

logging.basicConfig(level=logging.ERROR)

from app.core.aws import bedrock_runtime_client  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.models.schemas import SanitizedMessage, SummarizeRequest  # noqa: E402
from app.services.parsing import summary_from_response  # noqa: E402
from app.services.pipeline import SUMMARY_MAX_TOKENS, SummarizationPipeline  # noqa: E402
from app.services.prompts import SUMMARY_SYSTEM_PROMPT, summary_user_prompt  # noqa: E402

# (label, model id, additionalModelRequestFields)
CANDIDATES = [
    ("haiku-4.5", "us.anthropic.claude-haiku-4-5-20251001-v1:0", None),
    ("haiku-5.5 default", "us.anthropic.claude-haiku-5-5", None),
    ("haiku-5.5 effort=low", "us.anthropic.claude-haiku-5-5", {"output_config": {"effort": "low"}}),
    ("haiku-5.5 thinking off", "us.anthropic.claude-haiku-5-5", {"thinking": {"type": "disabled"}}),
    ("gpt-oss-20b", "openai.gpt-oss-20b-1:0", None),
    ("gpt-oss-120b", "openai.gpt-oss-120b-1:0", None),
    ("gpt-5.4", "us.openai.gpt-5.4", None),
    ("gpt-5.5", "us.openai.gpt-5.5", None),
    ("gpt-5.6-luna", "us.openai.gpt-5.6-luna", None),
    ("gpt-5.6-terra", "us.openai.gpt-5.6-terra", None),
    ("gpt-5.6-sol", "us.openai.gpt-5.6-sol", None),
    ("gpt-6-luna", "us.openai.gpt-6-luna", None),
    ("gpt-6-astra", "us.openai.gpt-6-astra", None),
    ("gpt-6-sol", "us.openai.gpt-6-sol", None),
    ("gpt-6.1-terra", "us.openai.gpt-6.1-terra", None),
    ("gpt-6.1-sol", "us.openai.gpt-6.1-sol", None),
]

settings = Settings()
client = bedrock_runtime_client(settings, sdk_retries=True)
request = SummarizeRequest.model_validate(json.load(open(sys.argv[1], encoding="utf-8")))
conversation = SummarizationPipeline._conversation(
    [
        SanitizedMessage(role=m.role, original_content="", sanitized_content=m.content, pii_detected=[], pii_count=0)
        for m in request.messages
    ]
)
prompt = summary_user_prompt(conversation, request.summary_style, request.language)


def screen(candidate):
    label, model_id, extra = candidate
    kwargs = dict(
        modelId=model_id,
        system=[{"text": SUMMARY_SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": SUMMARY_MAX_TOKENS},
    )
    if extra:
        kwargs["additionalModelRequestFields"] = extra
    try:
        started = time.perf_counter()
        response = client.converse(**kwargs)
        ms = (time.perf_counter() - started) * 1000
        blocks = response["output"]["message"]["content"]
        text = "\n".join(b["text"] for b in blocks if "text" in b)
        reasoning = any("reasoningContent" in b for b in blocks)
        return label, ms, response["usage"]["outputTokens"], bool(summary_from_response(text).strip()), reasoning, None
    except Exception as exc:  # noqa: BLE001
        return label, None, None, None, None, f"{type(exc).__name__}: {str(exc)[:150]}"


with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(screen, CANDIDATES))

print(f"{'model':24} {'latency':>8} {'out tok':>7} json reasoning")
for label, ms, tokens, ok, reasoning, err in sorted(results, key=lambda r: r[1] or 1e9):
    print(f"{label:24} ERROR {err}" if err else f"{label:24} {ms:>6.0f}ms {tokens:>7} {str(ok):>4} {reasoning}")
