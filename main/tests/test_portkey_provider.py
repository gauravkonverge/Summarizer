from types import SimpleNamespace

from app.core.config import Settings
from app.providers.portkey import PortkeyProvider


class FakePortkeyClient:
    def __init__(self):
        self.request = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="first\nsecond"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )


def test_portkey_chat_completions_mapping():
    client = FakePortkeyClient()
    provider = PortkeyProvider(
        Settings(
            llm_model_id="approved.model-v1",
            portkey_api_key="test-key",
            portkey_provider="@test-provider",
        ),
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
    assert client.request["model"] == "approved.model-v1"
    assert client.request["temperature"] == 0.2
    assert client.request["max_tokens"] == 100
    assert client.request["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_portkey_retries_on_rate_limit_then_succeeds():
    client = FakePortkeyClient()
    calls = {"count": 0}
    original_create = client._create

    def flaky_create(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            error = RuntimeError("rate limited")
            error.status_code = 429
            raise error
        return original_create(**kwargs)

    client.chat.completions.create = flaky_create

    provider = PortkeyProvider(
        Settings(
            llm_model_id="approved.model-v1",
            portkey_api_key="test-key",
            portkey_provider="@test-provider",
            llm_retry_base_delay_seconds=0.0,
            llm_retry_max_delay_seconds=0.0,
        ),
        client=client,
    )

    result = provider.complete(
        system_prompt="system",
        user_prompt="user",
        temperature=0.2,
        max_tokens=100,
    )

    assert calls["count"] == 2
    assert result.text == "first\nsecond"
