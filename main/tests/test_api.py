from app.core.config import Settings
from app.main import create_app
from app.models.schemas import SummarizeRequest
from tests.fakes import FakeProvider, FakeSanitizer


def test_health_and_summarize_api_without_aws_credentials():
    provider = FakeProvider()
    app = create_app(
        settings=Settings(
            bedrock_model_id=provider.model_id,
        ),
        provider=provider,
        sanitizer=FakeSanitizer(),
    )
    health_endpoint = next(route.endpoint for route in app.routes if route.path == "/health")
    summary_endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/summarize"
    )
    request = SummarizeRequest.model_validate(
        {
            "messages": [
                {"role": "customer", "content": "Please check john@example.com."},
                {"role": "support", "content": "The request is being checked."},
            ],
            "summary_style": "brief",
            "language": "en",
        }
    )

    assert health_endpoint() == {"status": "ok"}
    body = summary_endpoint(request).model_dump()
    assert body["model_used"] == provider.model_id
    assert body["total_pii_entities_removed"] == 1
    assert len(body["llm_call_inputs"]) == 2
