"""Standalone fake IdP for manually testing authentication against a running
`uvicorn app.main:app` instance. Not used by the app or by pytest — the
automated tests in tests/test_auth.py stub the JWKS client directly and need
no server at all. Use this only for end-to-end curl testing.

Usage:
    .venv/bin/python devtools/local_idp.py

It starts a JWKS HTTP server on localhost, prints the .env values that match
it, and mints a valid access token (plus a couple of intentionally-bad ones
for negative testing). Leave it running in one terminal while you run
uvicorn in another and curl against it in a third.
"""

import http.server
import json
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

JWKS_PORT = 9999
ISSUER = "http://localhost:9999/"
AUDIENCE = "api://summarizer-local-test"
SCOPE = "summarizer.invoke"
KEY_ID = "local-dev-key-1"

_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks_document() -> dict:
    jwk = RSAAlgorithm.to_jwk(_private_key.public_key(), as_dict=True)
    jwk.update({"kid": KEY_ID, "use": "sig", "alg": "RS256"})
    return {"keys": [jwk]}


class _JwksHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps(_jwks_document()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass  # keep stdout free for the printed tokens/instructions


def _mint(*, expires_in_minutes: float = 10, **claim_overrides) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=expires_in_minutes),
        "scp": SCOPE,
        "azp": "local-test-caller",
    }
    claims.update(claim_overrides)
    return jwt.encode(claims, _private_key, algorithm="RS256", headers={"kid": KEY_ID})


def main() -> None:
    server = http.server.HTTPServer(("127.0.0.1", JWKS_PORT), _JwksHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    jwks_url = f"http://localhost:{JWKS_PORT}/keys"

    valid_token = _mint()
    expired_token = _mint(expires_in_minutes=-5)
    wrong_scope_token = _mint(scp="something.else")

    print(f"JWKS server running at {jwks_url}  (Ctrl+C to stop)\n")
    print("Set these in main/.env, then restart uvicorn:")
    print("  AUTH_ENABLED=true")
    print(f"  OIDC_JWKS_URL={jwks_url}")
    print(f"  OIDC_ISSUER={ISSUER}")
    print(f"  OIDC_AUDIENCE={AUDIENCE}")
    print(f"  OIDC_REQUIRED_SCOPE={SCOPE}")
    print()
    print("Valid token (expect 200):")
    print(f"  export TOKEN='{valid_token}'")
    print()
    print("Expired token (expect 401):")
    print(f"  export EXPIRED_TOKEN='{expired_token}'")
    print()
    print("Wrong-scope token (expect 403):")
    print(f"  export WRONG_SCOPE_TOKEN='{wrong_scope_token}'")
    print()
    print("Example:")
    print("  curl -i -X POST http://localhost:8080/api/summarize \\")
    print("    -H \"Authorization: Bearer $TOKEN\" -H 'Content-Type: application/json' \\")
    print("    --data '{\"messages\":[{\"role\":\"customer\",\"content\":\"hi\"},")
    print("                          {\"role\":\"support\",\"content\":\"hello\"}]}'")
    print()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopping JWKS server.")
        server.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
