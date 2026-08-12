import asyncio
import json
import logging

from app.core.config import Settings
from app.core.observability import (
    JsonFormatter,
    RequestObservabilityMiddleware,
    cloudwatch_metric_event,
)
from app.main import create_app
from app.models.schemas import SummarizeRequest
from tests.fakes import FakeProvider, FakeSanitizer


def test_cloudwatch_metric_event_uses_only_bounded_dimensions():
    settings = Settings(
        app_env="local",
        cloudwatch_emf_enabled=True,
        cloudwatch_metrics_namespace="Corpay/Test",
    )

    event = cloudwatch_metric_event(
        settings,
        operation="Summarize",
        metrics=[
            ("RequestCount", 1, "Count"),
            ("RequestLatency", 12.5, "Milliseconds"),
        ],
        properties={"request_id": "unique-request", "status_code": 200},
    )

    directive = event["_aws"]["CloudWatchMetrics"][0]
    assert directive["Namespace"] == "Corpay/Test"
    assert directive["Dimensions"] == [["Environment", "Operation"]]
    assert event["RequestCount"] == 1
    assert event["request_id"] == "unique-request"
    assert "request_id" not in directive["Dimensions"][0]


def test_json_formatter_emits_one_valid_json_object():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="safe event\nwith newline",
        args=(),
        exc_info=None,
    )

    event = json.loads(formatter.format(record))

    assert event["level"] == "INFO"
    assert event["logger"] == "app.test"
    assert event["message"] == "safe event\nwith newline"


def test_request_middleware_adds_request_id_header():
    sent = []

    async def downstream(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestObservabilityMiddleware(
        downstream,
        settings=Settings(app_env="local", cloudwatch_emf_enabled=False),
    )

    async def invoke():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await middleware(
            {"type": "http", "path": "/health"},
            receive,
            send,
        )

    asyncio.run(invoke())

    headers = dict(sent[0]["headers"])
    assert sent[0]["status"] == 200
    assert len(headers[b"x-request-id"]) == 32


def test_summarize_endpoint_emits_aggregate_metrics(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        "app.api.routes.emit_cloudwatch_metrics",
        lambda _settings, **event: emitted.append(event),
    )
    provider = FakeProvider()
    app = create_app(
        settings=Settings(
            app_env="local",
            llm_model_id=provider.model_id,
            include_llm_call_inputs=False,
            cloudwatch_emf_enabled=True,
        ),
        provider=provider,
        sanitizer=FakeSanitizer(),
    )
    summary_endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/summarize"
    )
    request = SummarizeRequest.model_validate(
        {
            "messages": [
                {"role": "customer", "content": "Please check john@example.com."},
                {"role": "support", "content": "The request is being checked."},
            ]
        }
    )

    summary_endpoint(request, caller="test-client")

    metrics = {name: value for name, value, _unit in emitted[0]["metrics"]}
    assert metrics == {
        "SummarizationSuccess": 1,
        "InputTokens": 180,
        "OutputTokens": 45,
        "PIIEntityCount": 1,
    }
