"""Shared identity + decision types used across auth, policy and MCP layers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Caller:
    """Who is making a request, as resolved from the user JWT. Passed down to sub-agent
    graphs and across MCP calls (as plain dict _meta) so tools can do row-level scoping."""

    user_id: str
    team: str
    roles: tuple[str, ...] = field(default_factory=tuple)

    def to_meta(self) -> dict:
        return {"user_id": self.user_id, "team": self.team, "roles": list(self.roles)}

    @staticmethod
    def from_meta(meta: dict) -> "Caller":
        return Caller(user_id=meta["user_id"], team=meta["team"], roles=tuple(meta.get("roles", ())))

    def has_role(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str

    @staticmethod
    def allowed(reason: str = "ok") -> "Decision":
        return Decision(True, reason)

    @staticmethod
    def denied(reason: str) -> "Decision":
        return Decision(False, reason)
