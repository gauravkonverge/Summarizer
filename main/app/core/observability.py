"""PII-safe structured logging and CloudWatch Embedded Metric Format helpers."""

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Iterable
from uuid import uuid4

from app.core.config import Settings

Metric = tuple[str, int | float, str]

_emf_logger = logging.getLogger("cloudwatch.emf")
_emf_logger.propagate = False
if not _emf_logger.handlers:
    _emf_handler = logging.StreamHandler()
    _emf_handler.setFormatter(logging.Formatter("%(message)s"))
    _emf_logger.addHandler(_emf_handler)
_emf_logger.setLevel(logging.INFO)


class JsonFormatter(logging.Formatter):
    """Render one JSON object per line for CloudWatch Logs Insights."""

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, separators=(",", ":"), ensure_ascii=False)


def cloudwatch_metric_event(
    settings: Settings,
    *,
    operation: str,
    metrics: Iterable[Metric],
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bounded-dimension CloudWatch EMF event.

    Request IDs and diagnostic properties are included for Logs Insights only;
    they are deliberately excluded from metric dimensions.
    """

    metric_values = list(metrics)
    event: dict[str, Any] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": settings.cloudwatch_metrics_namespace,
                    "Dimensions": [["Environment", "Operation"]],
                    "Metrics": [
                        {"Name": name, "Unit": unit}
                        for name, _value, unit in metric_values
                    ],
                }
            ],
        },
        "Environment": settings.cloudwatch_environment,
        "Operation": operation,
    }
    event.update({name: value for name, value, _unit in metric_values})
    if properties:
        event.update(properties)
    return event


def emit_cloudwatch_metrics(
    settings: Settings,
    *,
    operation: str,
    metrics: Iterable[Metric],
    properties: dict[str, Any] | None = None,
) -> None:
    """Write an EMF event to stdout for collection by the CloudWatch Agent."""

    if not settings.cloudwatch_emf_enabled:
        return
    event = cloudwatch_metric_event(
        settings,
        operation=operation,
        metrics=metrics,
        properties=properties,
    )
    _emf_logger.info(json.dumps(event, separators=(",", ":"), ensure_ascii=False))


class RequestObservabilityMiddleware:
    """Pure ASGI request timing middleware with a correlation response header."""

    def __init__(self, app: Any, *, settings: Settings):
        self.app = app
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                message = dict(message)
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            path = str(scope.get("path", ""))
            operation = (
                "Summarize"
                if path == "/api/summarize"
                else "Health"
                if path == "/health"
                else "Other"
            )
            latency_ms = (time.perf_counter() - started) * 1000
            metrics: list[Metric] = [
                ("RequestCount", 1, "Count"),
                ("RequestLatency", round(latency_ms, 3), "Milliseconds"),
            ]
            if 400 <= status_code < 500:
                metrics.append(("Http4xxCount", 1, "Count"))
            elif status_code >= 500:
                metrics.append(("Http5xxCount", 1, "Count"))
            emit_cloudwatch_metrics(
                self.settings,
                operation=operation,
                metrics=metrics,
                properties={
                    "request_id": request_id,
                    "status_code": status_code,
                },
            )
            self.logger.info(
                "Completed request request_id=%s operation=%s status_code=%d latency_ms=%.3f",
                request_id,
                operation,
                status_code,
                latency_ms,
            )
