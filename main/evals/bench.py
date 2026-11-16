"""Benchmark the summarizer pipeline stages on one sample request (live AWS)."""

import dataclasses
import json
import logging
import sys
import time

sample_path = sys.argv[1]
logging.basicConfig(level=logging.WARNING)

from app.core.config import Settings  # noqa: E402  (loads .env)
from app.models.schemas import SummarizeRequest  # noqa: E402
from app.providers.bedrock import BedrockProvider  # noqa: E402
from app.services.guardrail import BedrockGuardrailSanitizer  # noqa: E402
from app.services.pipeline import SummarizationPipeline  # noqa: E402

settings = Settings()
request = SummarizeRequest.model_validate(json.load(open(sample_path, encoding="utf-8")))
contents = [m.content for m in request.messages]
sanitizer = BedrockGuardrailSanitizer(settings)
provider = BedrockProvider(settings)
sanitizer.warm_up()
provider.warm_up()
print(f"model={settings.bedrock_model_id} messages={len(contents)}")

# Warm the connection pool so neither run pays TLS setup alone.
sanitizer.sanitize(contents[0])

for concurrency in (1, 8):
    pipe = SummarizationPipeline(
        provider=provider,
        settings=dataclasses.replace(settings, guardrail_max_concurrency=concurrency),
        sanitizer=sanitizer,
    )
    started = time.perf_counter()
    pipe._sanitize_all(contents)
    print(f"guardrail concurrency={concurrency}: {(time.perf_counter() - started) * 1000:.0f} ms")

# One full end-to-end request; stage timings come from the pipeline's own log line.
logging.getLogger("app.services.pipeline").setLevel(logging.INFO)
pipe = SummarizationPipeline(provider=provider, settings=settings, sanitizer=sanitizer)
started = time.perf_counter()
response = pipe.summarize(request)
print(f"end-to-end: {(time.perf_counter() - started) * 1000:.0f} ms")
print(f"tokens in/out: {response.inference_cost.input_tokens}/{response.inference_cost.output_tokens}")
print(f"pii removed: {response.total_pii_entities_removed} types={response.unique_pii_types_found}")
print(f"confidence: {response.confidence}")
print("summary:\n" + response.summary)
