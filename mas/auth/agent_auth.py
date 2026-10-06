"""Agent <-> orchestrator authentication handshake.

Protocol (challenge-response, HMAC-SHA256 over a per-agent shared secret):
  1. Orchestrator issues a one-time nonce for an agent id.
  2. The agent computes HMAC-SHA256(secret, "<agent_id>:<nonce>") and sends it back.
  3. The orchestrator recomputes the HMAC with the secret *it* expects for that agent
     (from config/agents.yaml) and compares. On match it issues a short-lived agent
     session JWT and registers the agent as authenticated.

No sub-agent is ever invoked, and no sub-agent's `delegate_to_*` tool is ever offered to the
router, unless `AgentSessionRegistry.is_authenticated(agent_id)` is true. Sessions expire and
must be refreshed (see `ensure_authenticated`).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field

import jwt

from mas.auth.jwt_utils import jwt_secret
from mas.config import agents_config, settings_config


class AgentAuthError(Exception):
    pass


def agent_secret(agent_id: str) -> str:
    """The shared secret this agent is expected to know (env var, falling back to the
    prototype's dev default so the demo runs with no setup)."""
    acfg = agents_config()["agents"][agent_id]
    return os.environ.get(acfg["secret_env"], acfg["dev_secret"])


def compute_response(agent_id: str, nonce: str, secret: str) -> str:
    msg = f"{agent_id}:{nonce}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


@dataclass
class AgentSession:
    agent_id: str
    token: str
    expires_at: float


@dataclass
class AgentSessionRegistry:
    _sessions: dict[str, AgentSession] = field(default_factory=dict)
    _nonces: dict[str, str] = field(default_factory=dict)
    # Once an agent completes the handshake successfully, it stays in here for the life of
    # the registry - even after its session later expires. This is what lets
    # ensure_authenticated distinguish "expired, safe to silently refresh" from "never
    # completed the handshake in the first place", which must never silently self-heal just
    # because an LLM happened to call its delegate tool.
    _ever_authenticated: set[str] = field(default_factory=set)

    def issue_challenge(self, agent_id: str) -> str:
        nonce = secrets.token_hex(16)
        self._nonces[agent_id] = nonce
        return nonce

    def _verify_response(self, agent_id: str, response: str) -> bool:
        nonce = self._nonces.pop(agent_id, None)
        if nonce is None:
            return False
        expected = compute_response(agent_id, nonce, agent_secret(agent_id))
        return hmac.compare_digest(expected, response)

    def authenticate(self, agent_id: str, response: str) -> AgentSession | None:
        if not self._verify_response(agent_id, response):
            self._sessions.pop(agent_id, None)
            return None
        ttl = settings_config()["jwt"]["agent_token_ttl_seconds"]
        now = time.time()
        payload = {"sub": agent_id, "scope": "agent", "iat": now, "exp": now + ttl}
        token = jwt.encode(payload, jwt_secret(), algorithm="HS256")
        session = AgentSession(agent_id=agent_id, token=token, expires_at=now + ttl)
        self._sessions[agent_id] = session
        self._ever_authenticated.add(agent_id)
        return session

    def is_authenticated(self, agent_id: str) -> bool:
        s = self._sessions.get(agent_id)
        return s is not None and s.expires_at > time.time()

    def ever_authenticated(self, agent_id: str) -> bool:
        """True once this agent has completed the handshake at least once - stays True even
        after its session later expires (expiry is expected to self-heal via
        ensure_authenticated; never having authenticated at all must not)."""
        return agent_id in self._ever_authenticated

    def valid_token(self, agent_id: str) -> str | None:
        s = self._sessions.get(agent_id)
        if s is None or s.expires_at <= time.time():
            return None
        return s.token

    def authenticated_agent_ids(self) -> list[str]:
        now = time.time()
        return [aid for aid, s in self._sessions.items() if s.expires_at > now]

    def revoke(self, agent_id: str) -> None:
        self._sessions.pop(agent_id, None)


def run_handshake(registry: AgentSessionRegistry, agent_id: str, claimed_secret: str) -> AgentSession | None:
    """Drives both sides of the handshake for a single agent. `claimed_secret` is what the
    "agent" presents - in this single-process prototype that's the same value `agent_secret`
    resolves to for a real agent, or a wrong value in tests/misconfiguration."""
    nonce = registry.issue_challenge(agent_id)
    response = compute_response(agent_id, nonce, claimed_secret)
    return registry.authenticate(agent_id, response)


def ensure_authenticated(registry: AgentSessionRegistry, agent_id: str) -> str:
    """Returns a valid session token for agent_id, re-running the handshake if a
    previously-valid session expired. Deliberately does NOT perform a first-time handshake
    here - that only happens once, explicitly, at startup (see mas/runtime.py). An agent
    that never completed it stays excluded for the life of the process, even if something
    later tries to delegate to it anyway (e.g. an LLM hallucinating a tool call) - silently
    re-authenticating it here would defeat the "authenticate before any conversation" rule.
    Raises AgentAuthError in both cases: never authenticated, or re-authentication failed."""
    token = registry.valid_token(agent_id)
    if token is not None:
        return token
    if not registry.ever_authenticated(agent_id):
        raise AgentAuthError(f"agent '{agent_id}' has never completed its auth handshake")
    session = run_handshake(registry, agent_id, agent_secret(agent_id))
    if session is None:
        raise AgentAuthError(f"agent '{agent_id}' failed re-authentication")
    return session.token
