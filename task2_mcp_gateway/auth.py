"""Bearer-token verification. Tokens are HS256 JWTs carrying a `role` claim."""

import os
import time
from dataclasses import dataclass

import jwt

ROLES = {"admin", "viewer"}
DEFAULT_SECRET = "dev-only-secret-change-me-in-production-32b"


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str


def secret() -> str:
    return os.environ.get("GATEWAY_JWT_SECRET", DEFAULT_SECRET)


def authenticate(authorization: str | None, key: str) -> Principal:
    if not authorization:
        raise AuthError("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must be 'Bearer <token>'")

    try:
        claims = jwt.decode(
            token.strip(), key, algorithms=["HS256"], options={"require": ["sub", "role", "exp"]}
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from None

    role = claims["role"]
    if role not in ROLES:
        raise AuthError("unknown role")
    return Principal(subject=str(claims["sub"]), role=role)


def mint_token(subject: str, role: str, key: str, ttl_seconds: int = 3600) -> str:
    now = int(time.time())
    return jwt.encode({"sub": subject, "role": role, "iat": now, "exp": now + ttl_seconds}, key, algorithm="HS256")


if __name__ == "__main__":
    import sys

    role = sys.argv[1] if len(sys.argv) > 1 else "viewer"
    print(mint_token(f"demo-{role}", role, secret()))
