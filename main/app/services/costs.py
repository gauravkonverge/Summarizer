"""Provider usage aggregation and configurable cost estimation."""

from app.core.config import Settings
from app.models.schemas import CallInferenceCost, InferenceCost
from app.providers.base import TokenUsage


def call_cost(usage: TokenUsage, settings: Settings) -> CallInferenceCost:
    input_usd = usage.input_tokens / 1_000_000 * settings.input_cost_per_million_tokens_usd
    output_usd = usage.output_tokens / 1_000_000 * settings.output_cost_per_million_tokens_usd
    total_usd = input_usd + output_usd
    multiplier = settings.usd_to_gbp_exchange_rate * 100
    return CallInferenceCost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        input_cost_usd=round(input_usd, 6),
        output_cost_usd=round(output_usd, 6),
        total_cost_usd=round(total_usd, 6),
        input_cost_gbp_pence=round(input_usd * multiplier, 4),
        output_cost_gbp_pence=round(output_usd * multiplier, 4),
        total_cost_gbp_pence=round(total_usd * multiplier, 4),
    )


def total_cost(usage: TokenUsage, settings: Settings) -> InferenceCost:
    return InferenceCost(**call_cost(usage, settings).model_dump())
