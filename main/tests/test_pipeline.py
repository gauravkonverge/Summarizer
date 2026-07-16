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
