"""Human-review gating: decides whether a tool call must wait for a human before it runs.

Driven by the `tools.<name>.approval` entries in config/agents.yaml (`required` or
`conditional` + a named condition from mas/access/conditions.py). This is a second gate
*after* RBAC (mas/access/policy.py) - a tool can be allowed by role but still require
approval because it's risky or the specific call is over some threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mas.access.conditions import CONDITIONS
from mas.access.models import Caller
from mas.config import agents_config
from mas.store.document_store import DocumentStore


@dataclass(frozen=True)
class ApprovalDecision:
    required: bool
    reason: str = ""
    approver_roles: tuple[str, ...] = ()


async def needs_approval(
    agent_id: str, tool_name: str, args: dict[str, Any], store: DocumentStore, caller: Caller
) -> ApprovalDecision:
    tool_cfg = agents_config()["agents"].get(agent_id, {}).get("tools", {}).get(tool_name, {})
    approval = tool_cfg.get("approval")
    approver_roles = tuple(tool_cfg.get("approver_roles", []))

    if approval == "required":
        risk = tool_cfg.get("risk", "high")
        return ApprovalDecision(True, f"'{tool_name}' always requires approval (risk: {risk})", approver_roles)

    if approval == "conditional":
        condition_name = tool_cfg["condition"]
        condition = CONDITIONS[condition_name]
        if await condition(args, store, caller):
            return ApprovalDecision(True, f"'{tool_name}' call triggered the '{condition_name}' condition", approver_roles)

    return ApprovalDecision(False)
