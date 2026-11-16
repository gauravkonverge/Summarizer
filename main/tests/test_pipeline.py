import json
from pathlib import Path

from app.core.config import Settings
from app.models.schemas import SummarizeRequest
from app.services.pipeline import SummarizationPipeline
from tests.fakes import FakeProvider, FakeSanitizer


def test_pipeline_preserves_response_contract_and_combines_usage():
    provider = FakeProvider()
    settings = Settings(
        app_env="local",
        bedrock_model_id=provider.model_id,
        include_original_content=True,
        include_llm_call_inputs=True,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
    )
    pipeline = SummarizationPipeline(
        provider=provider,
        settings=settings,
        sanitizer=FakeSanitizer(),
    )
    request = SummarizeRequest.model_validate(
        {
            "messages": [
                {
                    "role": "customer",
                    "content": "Contact me at john@example.com about the order.",
                    "timestamp": "2026-07-13T10:00:00Z",
                    "sender": "Customer",
                },
                {
                    "role": "support",
                    "content": "We will provide an order update.",
                    "timestamp": "2026-07-13T10:01:00Z",
                    "sender": "Agent",
                },
            ],
            "summary_style": "brief",
            "language": "en",
        }
    )

    response = pipeline.summarize(request)

    assert response.model_used == provider.model_id
    assert response.sanitized_messages[0].sanitized_content.startswith("Contact me at [EMAIL]")
    assert response.total_pii_entities_removed == 1
    assert response.confidence == 0.83
    assert response.inference_cost.input_tokens == 180
    assert response.inference_cost.output_tokens == 45
    assert response.timeline_metrics is not None
    assert response.timeline_metrics.message_count == 2
    assert len(provider.calls) == 2


def test_large_repository_sample_runs_without_live_credentials():
    sample_path = Path(__file__).parents[2] / "data" / "input_from_dtm_pdf.json"
    request = SummarizeRequest.model_validate(json.loads(sample_path.read_text()))
    provider = FakeProvider()
    pipeline = SummarizationPipeline(
        provider=provider,
        settings=Settings(
            app_env="local",
            bedrock_model_id=provider.model_id,
        ),
        sanitizer=FakeSanitizer(),
    )

    response = pipeline.summarize(request)

    assert response.timeline_metrics is not None
    assert response.timeline_metrics.message_count == 27
    assert response.timeline_metrics.duration_days == 76
    assert response.timeline_metrics.support_message_count == 17
    assert response.timeline_metrics.customer_message_count == 10


def test_pipeline_hides_original_content_and_prompts_when_disabled():
    provider = FakeProvider()
    pipeline = SummarizationPipeline(
        provider=provider,
        settings=Settings(
            app_env="ec2",
            bedrock_model_id=provider.model_id,
            include_original_content=False,
            include_llm_call_inputs=False,
            log_sanitization_details=False,
        ),
        sanitizer=FakeSanitizer(),
    )
    request = SummarizeRequest.model_validate(
        {
            "messages": [
                {"role": "customer", "content": "Contact john@example.com."},
                {"role": "support", "content": "We will provide an update."},
            ],
            "summary_style": "brief",
            "language": "en",
        }
    )

    response = pipeline.summarize(request)

    assert response.sanitized_messages[0].original_content == ""
    assert response.sanitized_messages[0].sanitized_content == "Contact [EMAIL]"
    assert response.llm_call_inputs == []


def test_pipeline_flags_garbled_messages_and_masks_them_in_the_prompt():
    provider = FakeProvider()
    pipeline = SummarizationPipeline(
        provider=provider,
        settings=Settings(app_env="local", bedrock_model_id=provider.model_id),
        sanitizer=FakeSanitizer(),
    )
    request = SummarizeRequest.model_validate(
        {
            "messages": [
                {"role": "customer", "content": "vehicle ???? status ????"},
                {"role": "support", "content": "We will provide an order update."},
            ],
            "summary_style": "brief",
            "language": "en",
        }
    )

    response = pipeline.summarize(request)

    assert response.sanitized_messages[0].is_garbled
    assert response.sanitized_messages[0].garbled_reason
    assert not response.sanitized_messages[1].is_garbled
    assert response.garbled_message_count == 1
    assert response.confidence == 0.76
    assert "unreadable/garbled" in response.confidence_reasoning

    summary_prompt = provider.calls[0]["user_prompt"]
    assert "vehicle ????" not in summary_prompt
    assert "[unreadable message:" in summary_prompt


def _request(contents: list[str]) -> SummarizeRequest:
    return SummarizeRequest.model_validate(
        {
            "messages": [
                {
                    "role": "customer",
                    "content": content,
                    "timestamp": f"2026-07-13T10:{index:02d}:00Z",
                    "sender": "Customer",
                }
                for index, content in enumerate(contents)
            ],
            "summary_style": "brief",
            "language": "en",
        }
    )


def test_guardrail_calls_run_concurrently_and_preserve_message_order():
    import threading

    from app.services.guardrail import SanitizationResult

    contents = [f"message {index}" for index in range(6)]
    # The barrier only releases once 3 calls are in flight at the same time;
    # with sequential sanitization it would time out.
    barrier = threading.Barrier(3, timeout=5)

    class SlowSanitizer:
        def sanitize(self, text: str) -> SanitizationResult:
            barrier.wait()
            return SanitizationResult(sanitized_text=text.upper(), original_text=text)

    provider = FakeProvider()
    pipeline = SummarizationPipeline(
        provider=provider,
        settings=Settings(bedrock_model_id=provider.model_id, guardrail_max_concurrency=3),
        sanitizer=SlowSanitizer(),
    )

    response = pipeline.summarize(_request(contents))

    assert [m.sanitized_content for m in response.sanitized_messages] == [
        content.upper() for content in contents
    ]


def test_guardrail_failure_fails_closed_without_calling_the_model():
    import pytest

    from app.services.guardrail import SanitizationError

    class FailingSanitizer(FakeSanitizer):
        def sanitize(self, text):
            if text == "bad":
                raise SanitizationError("blocked")
            return super().sanitize(text)

    provider = FakeProvider()
    pipeline = SummarizationPipeline(
        provider=provider,
        settings=Settings(bedrock_model_id=provider.model_id, guardrail_max_concurrency=4),
        sanitizer=FailingSanitizer(),
    )

    with pytest.raises(SanitizationError):
        pipeline.summarize(_request(["ok", "bad", "ok", "ok"]))
    assert provider.calls == []


def test_stage_latency_metrics_are_emitted(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        "app.services.pipeline.emit_cloudwatch_metrics",
        lambda _settings, **event: emitted.append(event),
    )
    provider = FakeProvider()
    pipeline = SummarizationPipeline(
        provider=provider,
        settings=Settings(bedrock_model_id=provider.model_id),
        sanitizer=FakeSanitizer(),
    )

    pipeline.summarize(_request(["hello", "world"]))

    assert len(emitted) == 1
    event = emitted[0]
    assert event["operation"] == "Summarize"
    assert event["properties"] == {"message_count": 2}
    metrics = {name: (value, unit) for name, value, unit in event["metrics"]}
    assert set(metrics) == {
        "GuardrailLatency",
        "SummaryLatency",
        "VerifierLatency",
        "SummaryFallbackCount",
    }
    assert metrics["SummaryFallbackCount"] == (0, "Count")
    for name in ("GuardrailLatency", "SummaryLatency", "VerifierLatency"):
        value, unit = metrics[name]
        assert value >= 0 and unit == "Milliseconds"
