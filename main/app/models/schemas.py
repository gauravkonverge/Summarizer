"""Pydantic request and response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "system", "agent", "customer", "support"]
    content: str = Field(..., min_length=1)
    timestamp: str | None = Field(None, description="ISO-8601 timestamp")
    sender: str | None = Field(None, description="Display name of sender")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class SummarizeRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=1000)
    summary_style: Literal["brief", "detailed", "bullet_points"] = "detailed"
    language: str = Field(default="en", min_length=2, max_length=10, pattern=r"^[A-Za-z-]+$")


class SanitizedMessage(BaseModel):
    role: str
    original_content: str
    sanitized_content: str
    pii_detected: list[str]
    pii_count: int


class CallInferenceCost(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    input_cost_gbp_pence: float
    output_cost_gbp_pence: float
    total_cost_gbp_pence: float


class InferenceCost(CallInferenceCost):
    pass


class InferenceCostBreakdown(BaseModel):
    summary_call: CallInferenceCost = Field(
        ..., description="Summary-generation LLM call"
    )
    verifier_call: CallInferenceCost
    total_both_calls: CallInferenceCost


class TimelineMetrics(BaseModel):
    duration_days: int
    message_count: int
    start_date: str
    end_date: str
    support_message_count: int
    customer_message_count: int
    unique_senders: int


class LLMCallInput(BaseModel):
    call_name: str
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    user_prompt: str


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    summary: str
    confidence: float = Field(..., ge=0, le=1)
    confidence_reasoning: str
    model_used: str
    sanitized_messages: list[SanitizedMessage]
    total_pii_entities_removed: int
    unique_pii_types_found: list[str]
    inference_cost: InferenceCost
    inference_cost_breakdown: InferenceCostBreakdown
    timeline_metrics: TimelineMetrics | None = None
    llm_call_inputs: list[LLMCallInput] = Field(
        default_factory=list,
        description="Provider-neutral LLM prompt inputs",
    )
