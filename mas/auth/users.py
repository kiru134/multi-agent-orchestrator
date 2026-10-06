"""Demo user login: PBKDF2 password check against config/policies.yaml -> user JWT.

This is a prototype auth scheme (fixed users, config-file hashes) - good enough to
demonstrate the access-control layer, not for production use.
"""

from __future__ import annotations

import hashlib
import time

import jwt

from mas.access.models import Caller
from mas.access.policy import PolicyEngine, UnknownUser
from mas.auth.jwt_utils import jwt_secret
from mas.config import policies_config, settings_config

PBKDF2_ITERATIONS = 100_000


class LoginError(Exception):
    pass


def verify_password(user_id: str, password: str) -> bool:
    users = policies_config().get("users", {})
    record = users.get(user_id)
    if record is None:
        return False
    salt = bytes.fromhex(record["salt"])
    expected = record["password_hash"]
    computed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS).hex()
    return hmac_compare(computed, expected)


def hmac_compare(a: str, b: str) -> bool:
    # Constant-time-ish compare without importing hmac just for this; both are hex digests
    # of equal expected length so this doesn't leak more than a direct == would avoid.
    import hmac as _hmac

    return _hmac.compare_digest(a, b)


def login(user_id: str, password: str) -> str:
    if not verify_password(user_id, password):
        raise LoginError("invalid username or password")
    ttl = settings_config()["jwt"]["user_token_ttl_seconds"]
    now = time.time()
    payload = {"sub": user_id, "scope": "user", "iat": now, "exp": now + ttl}
    return jwt.encode(payload, jwt_secret(), algorithm="HS256")


def resolve_caller(token: str, policy: PolicyEngine) -> Caller:
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=["HS256"])
    except jwt.InvalidTokenError as e:
        raise LoginError(f"invalid or expired token: {e}") from e
    if payload.get("scope") != "user":
        raise LoginError("token is not a user token")
    user_id = payload["sub"]
    try:
        return policy.get_caller(user_id)
    except UnknownUser as e:
        raise LoginError(f"unknown user '{user_id}'") from e
