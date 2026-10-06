"""RBAC: agent-level and tool-level access checks, plus an in-memory audit log.

Two checks matter here:
  - can_use_agent(caller, agent_id)  -> used by the orchestrator's delegate node
  - can_call_tool(caller, tool_name) -> used by each sub-agent's authorize_tools node

Both are driven by config/policies.yaml (roles -> agents/tools, users -> roles/team).
Tool patterns support "it.*" and "*" wildcards via fnmatch.
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from typing import Any

from mas.access.models import Caller, Decision
from mas.config import policies_config


class UnknownUser(Exception):
    pass


@dataclass
class AuditEntry:
    ts: float
    user_id: str
    kind: str  # "agent" | "tool"
    resource: str
    allowed: bool
    reason: str


@dataclass
class PolicyEngine:
    _audit_log: list[AuditEntry] = field(default_factory=list)

    # ---- user profile lookups (roles/team come from the same config) ----

    def _users(self) -> dict[str, Any]:
        return policies_config().get("users", {})

    def _roles(self) -> dict[str, Any]:
        return policies_config().get("roles", {})

    def get_caller(self, user_id: str) -> Caller:
        users = self._users()
        if user_id not in users:
            raise UnknownUser(user_id)
        u = users[user_id]
        return Caller(user_id=user_id, team=u["team"], roles=tuple(u.get("roles", ())))

    def user_record(self, user_id: str) -> dict[str, Any] | None:
        return self._users().get(user_id)

    def all_user_ids(self) -> list[str]:
        return list(self._users().keys())

    # ---- checks ------------------------------------------------------

    def _role_defs(self, caller: Caller) -> list[dict[str, Any]]:
        roles = self._roles()
        return [roles[r] for r in caller.roles if r in roles]

    def can_use_agent(self, caller: Caller, agent_id: str) -> Decision:
        allowed_agents: set[str] = set()
        for role in self._role_defs(caller):
            allowed_agents.update(role.get("agents", []))
        decision = (
            Decision.allowed(f"role grants access to agent '{agent_id}'")
            if agent_id in allowed_agents
            else Decision.denied(f"no role of {list(caller.roles)} grants access to agent '{agent_id}'")
        )
        self._record(caller, "agent", agent_id, decision)
        return decision

    def can_call_tool(self, caller: Caller, tool_name: str) -> Decision:
        patterns: set[str] = set()
        for role in self._role_defs(caller):
            patterns.update(role.get("tools", []))
        matched = next((p for p in patterns if fnmatch.fnmatch(tool_name, p)), None)
        decision = (
            Decision.allowed(f"matched pattern '{matched}'")
            if matched
            else Decision.denied(f"no role of {list(caller.roles)} grants tool '{tool_name}'")
        )
        self._record(caller, "tool", tool_name, decision)
        return decision

    def is_approver(self, caller: Caller, approver_roles: list[str]) -> bool:
        return caller.has_role(*approver_roles)

    # ---- audit ---------------------------------------------------------

    def _record(self, caller: Caller, kind: str, resource: str, decision: Decision) -> None:
        self._audit_log.append(
            AuditEntry(
                ts=time.time(),
                user_id=caller.user_id,
                kind=kind,
                resource=resource,
                allowed=decision.allow,
                reason=decision.reason,
            )
        )

    def audit_log(self) -> list[AuditEntry]:
        return list(self._audit_log)


_default_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = PolicyEngine()
    return _default_engine
