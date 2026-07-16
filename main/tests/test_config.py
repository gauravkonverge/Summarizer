import pytest

from app.core.config import Settings


_LOCAL_AWS_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
)


def _remove_local_aws_credentials(monkeypatch):
    for name in _LOCAL_AWS_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_ec2_runtime_accepts_safe_configuration(monkeypatch):
    _remove_local_aws_credentials(monkeypatch)
    settings = Settings(
        app_env="ec2",
        include_original_content=False,
        include_llm_call_inputs=False,
        log_sanitization_details=False,
    )

    settings.validate_runtime()


def test_ec2_runtime_rejects_local_aws_credentials(monkeypatch):
    _remove_local_aws_credentials(monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic-test-key")
    settings = Settings(
        app_env="ec2",
        include_original_content=False,
        include_llm_call_inputs=False,
        log_sanitization_details=False,
    )

    with pytest.raises(RuntimeError, match="IAM role"):
        settings.validate_runtime()


@pytest.mark.parametrize(
    ("setting_name", "overrides"),
    [
        ("INCLUDE_ORIGINAL_CONTENT", {"include_original_content": True}),
        ("INCLUDE_LLM_CALL_INPUTS", {"include_llm_call_inputs": True}),
        ("LOG_SANITIZATION_DETAILS", {"log_sanitization_details": True}),
    ],
)
def test_ec2_runtime_rejects_unsafe_observability_settings(
    monkeypatch, setting_name, overrides
):
    _remove_local_aws_credentials(monkeypatch)
    values = {
        "app_env": "ec2",
        "include_original_content": False,
        "include_llm_call_inputs": False,
        "log_sanitization_details": False,
        **overrides,
    }

    with pytest.raises(RuntimeError, match=setting_name):
        Settings(**values).validate_runtime()


def test_runtime_rejects_unknown_environment():
    with pytest.raises(RuntimeError, match="APP_ENV"):
        Settings(app_env="production").validate_runtime()


def test_runtime_environment_is_case_insensitive():
    Settings(app_env="LOCAL").validate_runtime()
