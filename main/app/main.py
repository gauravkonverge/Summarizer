"""FastAPI application factory and default AWS Bedrock application."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import build_router
from app.core.config import Settings
from app.core.observability import RequestObservabilityMiddleware
from app.core.security import Authenticator, build_authenticator
from app.providers.base import LLMProvider
from app.providers.bedrock import BedrockProvider
from app.services.guardrail import BedrockGuardrailSanitizer, Sanitizer
from app.services.pipeline import SummarizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _warm_up(*components: object) -> None:
    """Create AWS clients at startup so the first request does not pay for it."""
    for component in components:
        warm_up = getattr(component, "warm_up", None)
        if warm_up is None:
            continue
        try:
            warm_up()
        except Exception:
            # Not fatal: the client is created lazily on first use instead.
            logger.warning("Startup warm-up failed for %s", type(component).__name__, exc_info=True)


def create_app(
    *,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    sanitizer: Sanitizer | None = None,
    authenticator: Authenticator | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_settings.validate_runtime()
    resolved_authenticator = authenticator or build_authenticator(resolved_settings)
    resolved_provider = provider or BedrockProvider(resolved_settings)
    resolved_sanitizer = sanitizer or BedrockGuardrailSanitizer(resolved_settings)
    pipeline = SummarizationPipeline(
        provider=resolved_provider,
        settings=resolved_settings,
        sanitizer=resolved_sanitizer,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _warm_up(resolved_provider, resolved_sanitizer)
        yield

    application = FastAPI(
        lifespan=lifespan,
        title=resolved_settings.app_name,
        description="PII-safe conversation summarization through AWS Bedrock Guardrails.",
        version="2.0.0",
    )

    application.add_middleware(
        RequestObservabilityMiddleware,
        settings=resolved_settings,
    )
    cors_origins = resolved_settings.cors_allowed_origin_list()
    if cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
    application.include_router(
        build_router(pipeline, resolved_authenticator, resolved_settings)
    )
    return application


app = create_app()
