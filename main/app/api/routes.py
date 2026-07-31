"""HTTP routes and safe exception translation."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse

from app.core.security import Authenticator
from app.core.config import Settings
from app.core.observability import emit_cloudwatch_metrics
from app.models.schemas import SummarizeRequest, SummarizeResponse
from app.providers.base import LLMProviderError
from app.services.guardrail import SanitizationError
from app.services.pipeline import SummarizationPipeline

logger = logging.getLogger(__name__)


def build_router(
    pipeline: SummarizationPipeline,
    authenticator: Authenticator,
    settings: Settings,
) -> APIRouter:
    router = APIRouter()

    # Unauthenticated by design: load balancer health checks carry no token.
    @router.get("/health", tags=["Health"])
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/docs/swagger/index", include_in_schema=False)
    def swagger_redirect() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @router.post(
        "/api/summarize",
        response_model=SummarizeResponse,
        status_code=status.HTTP_200_OK,
        tags=["Summarization"],
        summary="Sanitize PII and summarize conversation messages",
    )
    def summarize(
        request: SummarizeRequest,
        caller: str = Depends(authenticator),
    ) -> SummarizeResponse:
        # Identity only; message content is never logged here.
        logger.info(
            "Accepted summarize request caller=%s message_count=%d",
            caller,
            len(request.messages),
        )
        try:
            result = pipeline.summarize(request)
            emit_cloudwatch_metrics(
                settings,
                operation="Summarize",
                metrics=[
                    ("SummarizationSuccess", 1, "Count"),
                    (
                        "InputTokens",
                        result.inference_cost.input_tokens,
                        "Count",
                    ),
                    ("OutputTokens", result.inference_cost.output_tokens, "Count"),
                    (
                        "PIIEntityCount",
                        result.total_pii_entities_removed,
                        "Count",
                    ),
                ],
                properties={"message_count": len(request.messages)},
            )
            return result
        except SanitizationError as exc:
            emit_cloudwatch_metrics(
                settings,
                operation="Summarize",
                metrics=[("GuardrailFailure", 1, "Count")],
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except LLMProviderError as exc:
            metric_name = "BedrockThrottle" if exc.rate_limited else "BedrockFailure"
            emit_cloudwatch_metrics(
                settings,
                operation="Summarize",
                metrics=[(metric_name, 1, "Count")],
            )
            code = (
                status.HTTP_429_TOO_MANY_REQUESTS
                if exc.rate_limited
                else status.HTTP_502_BAD_GATEWAY
            )
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    return router
