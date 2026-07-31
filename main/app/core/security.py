"""Caller authentication for the summarization API.

Two authenticators share one contract: a FastAPI dependency that returns the
caller's identity string, or raises an HTTP error. The identity is logged with
each accepted request so PII-bearing inference can be attributed to a client.

`JwtAuthenticator` validates OAuth2 client-credentials tokens issued by the
organisation's IdP (Entra ID, Okta, or Ping). Token lifetime, rotation, and
revocation stay with the IdP; this service holds no caller secrets.
"""

import logging
from typing import Any, Protocol

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, PyJWKClientError

try:  # PyJWT >= 2.8 distinguishes JWKS transport failures from key errors.
    from jwt import PyJWKClientConnectionError
except ImportError:  # pragma: no cover - older PyJWT
    PyJWKClientConnectionError = ()  # type: ignore[assignment, misc]

from app.core.config import Settings

logger = logging.getLogger(__name__)

_UNAUTHENTICATED_HEADERS = {"WWW-Authenticate": "Bearer"}

# Claims each supported IdP uses to name the calling application.
_CALLER_CLAIMS = ("azp", "appid", "client_id", "cid", "sub")


class Authenticator(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> str: ...


def _unauthenticated(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=_UNAUTHENTICATED_HEADERS,
    )


class DisabledAuthenticator:
    """Used when AUTH_ENABLED is false. Rejected on EC2 by validate_runtime."""

    identity = "auth-disabled"

    def __call__(self) -> str:
        return self.identity


class JwtAuthenticator:
    """Validates a bearer JWT against the IdP's published signing keys."""

    def __init__(self, settings: Settings, jwk_client: Any | None = None):
        self.settings = settings
        self._jwk_client = jwk_client
        self._algorithms = settings.oidc_allowed_algorithm_list()
        self._required_scopes = settings.oidc_required_scope_list()

    def _get_jwk_client(self) -> Any:
        if self._jwk_client is None:
            self._jwk_client = PyJWKClient(
                self.settings.oidc_jwks_url,
                cache_keys=True,
                lifespan=self.settings.oidc_jwks_cache_seconds,
            )
        return self._jwk_client

    @staticmethod
    def _granted_scopes(claims: dict[str, Any]) -> set[str]:
        # `scope`/`scp` cover Okta and Ping; Entra ID application permissions
        # arrive in `roles`. Either string or list form is accepted.
        granted: set[str] = set()
        for name in ("scope", "scp", "roles"):
            value = claims.get(name)
            if isinstance(value, str):
                granted.update(value.split())
            elif isinstance(value, list):
                granted.update(str(item) for item in value if str(item).strip())
        return granted

    @staticmethod
    def _caller(claims: dict[str, Any]) -> str:
        for name in _CALLER_CLAIMS:
            value = claims.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "unidentified-caller"

    def _claims(self, token: str) -> dict[str, Any]:
        try:
            signing_key = self._get_jwk_client().get_signing_key_from_jwt(token)
        except PyJWKClientConnectionError as exc:
            # The IdP is unreachable. This is our dependency failing, not a bad
            # caller, so report it the same way a Guardrail outage is reported.
            logger.error("IdP JWKS endpoint unreachable (%s)", type(exc).__name__, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Token signing keys are temporarily unavailable.",
            ) from exc
        except PyJWKClientError as exc:
            logger.warning("No usable signing key for presented token (%s)", type(exc).__name__)
            raise _unauthenticated("Token signing key is not recognised.") from exc

        try:
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=self._algorithms,
                audience=self.settings.oidc_audience,
                issuer=self.settings.oidc_issuer,
                leeway=self.settings.oidc_clock_skew_seconds,
                options={"require": ["exp", "iat", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise _unauthenticated("Access token has expired.") from exc
        except jwt.InvalidTokenError as exc:
            # Covers bad signature, wrong audience/issuer, missing claims, and
            # disallowed algorithms. The reason is logged but not returned.
            logger.warning("Access token rejected (%s)", type(exc).__name__)
            raise _unauthenticated("Access token is not valid.") from exc

    def __call__(
        self,
        credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
    ) -> str:
        if credentials is None or not credentials.credentials.strip():
            raise _unauthenticated("An IdP-issued bearer token is required.")
        if credentials.scheme.lower() != "bearer":
            raise _unauthenticated("Authorization scheme must be Bearer.")

        claims = self._claims(credentials.credentials.strip())
        caller = self._caller(claims)

        missing = self._required_scopes - self._granted_scopes(claims)
        if missing:
            # Authenticated but not entitled, so 403 rather than 401.
            logger.warning("Caller %s is missing required scopes", caller)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access token does not grant the required scope.",
            )
        return caller


def build_authenticator(settings: Settings) -> Authenticator:
    settings.validate_for_auth()
    if not settings.auth_enabled:
        logger.warning(
            "AUTH_ENABLED is false; /api/summarize is unauthenticated. "
            "This is only permitted when APP_ENV=local."
        )
        return DisabledAuthenticator()
    return JwtAuthenticator(settings)
