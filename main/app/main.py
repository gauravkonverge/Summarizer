"""FastAPI application factory and default AWS Bedrock application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import build_router
from app.core.config import Settings
from app.providers.base import LLMProvider
from app.providers.bedrock import BedrockProvider
from app.services.guardrail import BedrockGuardrailSanitizer, Sanitizer
from app.services.pipeline import SummarizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def create_app(
    *,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    sanitizer: Sanitizer | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_provider = provider or BedrockProvider(resolved_settings)
    resolved_sanitizer = sanitizer or BedrockGuardrailSanitizer(resolved_settings)
    pipeline = SummarizationPipeline(
        provider=resolved_provider,
        settings=resolved_settings,
        sanitizer=resolved_sanitizer,
    )
    application = FastAPI(
        title=resolved_settings.app_name,
        description="PII-safe conversation summarization through AWS Bedrock Guardrails.",
        version="2.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(build_router(pipeline))
    return application


app = create_app()
