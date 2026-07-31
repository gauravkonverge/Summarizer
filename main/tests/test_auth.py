"""Caller authentication tests.

Tokens are signed with a locally generated RSA key and the JWKS client is
replaced with a stub, so no IdP or network access is required.
"""

from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
import jwt
from jwt import PyJWKClientError
import pytest

from app.core.config import Settings
from app.core.security import JwtAuthenticator, build_authenticator
from app.main import create_app
from tests.fakes import FakeProvider, FakeSanitizer

ISSUER = "https://idp.example.com/tenant-id/v2.0"
AUDIENCE = "api://summarizer"
SCOPE = "summarizer.invoke"

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class StubJwkClient:
    """Returns the local public key regardless of the token `kid`."""

    class _Key:
        key = _PRIVATE_KEY.public_key()

    def get_signing_key_from_jwt(self, token: str):
        return self._Key()


class UnreachableJwkClient:
    def get_signing_key_from_jwt(self, token: str):
        raise PyJWKClientError("Unable to find a signing key")


def auth_settings(**overrides) -> Settings:
    defaults = {
        "app_env": "local",
        "auth_enabled": True,
        "oidc_jwks_url": "https://idp.example.com/keys",
        "oidc_issuer": ISSUER,
        "oidc_audience": AUDIENCE,
        "oidc_required_scope": SCOPE,
    }
    return Settings(**{**defaults, **overrides})


def make_token(**overrides) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "scp": SCOPE,
        "azp": "caller-app-id",
    }
    claims.update(overrides)
    return jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256")


def build_client(settings: Settings, jwk_client=None) -> TestClient:
    authenticator = JwtAuthenticator(settings, jwk_client=jwk_client or StubJwkClient())
    app = create_app(
        settings=settings,
        provider=FakeProvider(),
        sanitizer=FakeSanitizer(),
        authenticator=authenticator,
    )
    return TestClient(app)


PAYLOAD = {
    "messages": [
        {"role": "customer", "content": "Please check john@example.com."},
        {"role": "support", "content": "The request is being checked."},
    ],
    "summary_style": "brief",
}


def post(client: TestClient, token: str | None = None, scheme: str = "Bearer"):
    headers = {"Authorization": f"{scheme} {token}"} if token else {}
    return client.post("/api/summarize", json=PAYLOAD, headers=headers)


def test_valid_token_is_accepted():
    response = post(build_client(auth_settings()), make_token())
    assert response.status_code == 200
    assert response.json()["summary"]


def test_missing_token_is_rejected():
    response = post(build_client(auth_settings()))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_non_bearer_scheme_is_rejected():
    response = post(build_client(auth_settings()), make_token(), scheme="Basic")
    assert response.status_code == 401


def test_expired_token_is_rejected():
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    token = make_token(iat=stale, exp=stale + timedelta(minutes=5))
    assert post(build_client(auth_settings()), token).status_code == 401


def test_wrong_audience_is_rejected():
    token = make_token(aud="api://some-other-service")
    assert post(build_client(auth_settings()), token).status_code == 401


def test_wrong_issuer_is_rejected():
    token = make_token(iss="https://attacker.example.com/v2.0")
    assert post(build_client(auth_settings()), token).status_code == 401


def test_token_signed_by_another_key_is_rejected():
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "scp": SCOPE,
        },
        other_key,
        algorithm="RS256",
    )
    assert post(build_client(auth_settings()), token).status_code == 401


def test_token_without_required_scope_is_forbidden():
    token = make_token(scp="some.other.scope")
    response = post(build_client(auth_settings()), token)
    assert response.status_code == 403
    assert "scope" in response.json()["detail"]


def test_entra_style_roles_claim_satisfies_scope():
    token = make_token(scp=None, roles=[SCOPE])
    assert post(build_client(auth_settings()), token).status_code == 200


def test_space_delimited_scope_claim_is_split():
    token = make_token(scp=None, scope=f"openid {SCOPE}")
    assert post(build_client(auth_settings()), token).status_code == 200


def test_unknown_signing_key_is_rejected_not_a_server_error():
    response = post(
        build_client(auth_settings(), jwk_client=UnreachableJwkClient()),
        make_token(),
    )
    assert response.status_code == 401


def test_health_endpoint_needs_no_token():
    assert build_client(auth_settings()).get("/health").status_code == 200


def test_missing_exp_claim_is_rejected():
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "iat": now, "scp": SCOPE},
        _PRIVATE_KEY,
        algorithm="RS256",
    )
    assert post(build_client(auth_settings()), token).status_code == 401


def test_empty_required_scope_accepts_any_valid_token():
    settings = auth_settings(oidc_required_scope="")
    token = make_token(scp=None)
    assert post(build_client(settings), token).status_code == 200


def test_auth_disabled_locally_allows_requests():
    settings = Settings(app_env="local", auth_enabled=False)
    app = create_app(
        settings=settings,
        provider=FakeProvider(),
        sanitizer=FakeSanitizer(),
    )
    assert TestClient(app).post("/api/summarize", json=PAYLOAD).status_code == 200


class TestAuthConfiguration:
    @pytest.fixture(autouse=True)
    def _no_local_aws_credentials(self, monkeypatch):
        # The EC2 runtime check reads these from the live environment.
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
            monkeypatch.delenv(name, raising=False)

    def test_auth_must_be_enabled_on_ec2(self):
        settings = Settings(app_env="ec2", auth_enabled=False, **_ec2_safe_flags())
        with pytest.raises(RuntimeError, match="AUTH_ENABLED must be true"):
            settings.validate_runtime()

    def test_wildcard_cors_rejected_on_ec2(self):
        settings = Settings(
            app_env="ec2", auth_enabled=True, cors_allowed_origins="*", **_ec2_safe_flags()
        )
        with pytest.raises(RuntimeError, match="CORS_ALLOWED_ORIGINS"):
            settings.validate_runtime()

    def test_ec2_accepts_authenticated_hardened_configuration(self):
        settings = Settings(
            app_env="ec2", auth_enabled=True, cors_allowed_origins="", **_ec2_safe_flags()
        )
        settings.validate_runtime()

    @pytest.mark.parametrize(
        "overrides, expected",
        [
            ({"oidc_jwks_url": ""}, "OIDC_JWKS_URL"),
            ({"oidc_issuer": ""}, "OIDC_ISSUER"),
            ({"oidc_audience": ""}, "OIDC_AUDIENCE"),
            ({"oidc_allowed_algorithms": ""}, "at least one algorithm"),
            ({"oidc_allowed_algorithms": "HS256"}, "asymmetric"),
            ({"oidc_allowed_algorithms": "RS256,none"}, "asymmetric"),
        ],
    )
    def test_incomplete_or_unsafe_auth_config_is_rejected(self, overrides, expected):
        with pytest.raises(RuntimeError, match=expected):
            auth_settings(**overrides).validate_for_auth()

    def test_disabled_auth_skips_oidc_validation(self):
        Settings(app_env="local", auth_enabled=False).validate_for_auth()

    def test_build_authenticator_returns_jwt_authenticator_when_enabled(self):
        assert isinstance(build_authenticator(auth_settings()), JwtAuthenticator)


def _ec2_safe_flags() -> dict:
    return {
        "include_original_content": False,
        "include_llm_call_inputs": False,
        "log_sanitization_details": False,
    }
