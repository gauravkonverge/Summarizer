"""HTTP routes and safe exception translation."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse

from app.models.schemas import SummarizeRequest, SummarizeResponse
from app.providers.base import LLMProviderError
from app.services.guardrail import SanitizationError
from app.services.pipeline import SummarizationPipeline


def build_router(pipeline: SummarizationPipeline) -> APIRouter:
    router = APIRouter()

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
    def summarize(request: SummarizeRequest) -> SummarizeResponse:
        try:
            return pipeline.summarize(request)
        except SanitizationError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except LLMProviderError as exc:
            code = (
                status.HTTP_429_TOO_MANY_REQUESTS
                if exc.rate_limited
                else status.HTTP_502_BAD_GATEWAY
            )
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    return router
