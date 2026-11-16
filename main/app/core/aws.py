"""Shared boto3 client construction for Bedrock Runtime."""

from typing import Any

from app.core.config import Settings


def bedrock_runtime_client(settings: Settings, *, sdk_retries: bool) -> Any:
    """Build a Bedrock Runtime client tuned for concurrent, latency-sensitive use.

    ``sdk_retries=False`` disables botocore's own retries for callers that
    implement their own retry loop, so throttles are not retried twice over
    (botocore attempts nested inside application attempts).
    """
    import boto3
    from botocore.config import Config

    config = Config(
        # Default is 10; parallel guardrail calls across concurrent requests
        # would otherwise queue waiting for a pooled connection.
        max_pool_connections=settings.aws_max_pool_connections,
        connect_timeout=5,
        read_timeout=60,
        tcp_keepalive=True,
        retries=(
            {"mode": "standard", "total_max_attempts": 3}
            if sdk_retries
            else {"mode": "standard", "total_max_attempts": 1}
        ),
    )
    kwargs: dict[str, Any] = {
        "service_name": "bedrock-runtime",
        "region_name": settings.aws_region,
        "config": config,
    }
    if settings.bedrock_endpoint_url:
        kwargs["endpoint_url"] = settings.bedrock_endpoint_url
    return boto3.client(**kwargs)
