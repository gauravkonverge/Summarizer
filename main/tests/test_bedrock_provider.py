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
