from app.core.config import Settings
from app.providers.bedrock import BedrockProvider


class FakeBedrockClient:
    def __init__(self):
        self.request = None

    def converse(self, **kwargs):
        self.request = kwargs
        return {
            "output": {"message": {"content": [{"text": "first"}, {"text": "second"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
        }


def test_bedrock_converse_mapping():
    client = FakeBedrockClient()
    provider = BedrockProvider(
        Settings(bedrock_model_id="approved.model-v1", aws_region="eu-west-2"),
        client=client,
    )

    result = provider.complete(
        system_prompt="system",
        user_prompt="user",
        temperature=0.2,
        max_tokens=100,
    )

    assert result.text == "first\nsecond"
    assert result.usage.total_tokens == 14
    assert client.request["modelId"] == "approved.model-v1"
    assert client.request["inferenceConfig"] == {"temperature": 0.2, "maxTokens": 100}


def test_bedrock_omits_temperature_when_not_set():
    """Claude Haiku 5.5 and newer models reject sampling parameters."""
    client = FakeBedrockClient()
    provider = BedrockProvider(
        Settings(bedrock_model_id="us.anthropic.claude-haiku-5-5", aws_region="us-east-1"),
        client=client,
    )

    provider.complete(system_prompt="system", user_prompt="user", max_tokens=100)

    assert client.request["inferenceConfig"] == {"maxTokens": 100}


def test_bedrock_clients_disable_sdk_retries_only_for_the_llm_provider():
    from app.core.aws import bedrock_runtime_client

    settings = Settings(aws_region="us-east-2", aws_max_pool_connections=42)
    llm = bedrock_runtime_client(settings, sdk_retries=False)
    guardrail = bedrock_runtime_client(settings, sdk_retries=True)

    assert llm.meta.config.max_pool_connections == 42
    assert llm.meta.config.retries["total_max_attempts"] == 1
    assert guardrail.meta.config.retries["total_max_attempts"] == 3
