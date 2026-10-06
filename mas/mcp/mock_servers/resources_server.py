"""Mock Resource Allocation MCP server: quotas, VM requests, laptop requests.

Quota tools are scoped to the caller's own team unless the caller has the manager role.
Approval/over-quota decisions are made upstream (the sub-agent graph's human_review node,
driven by config/agents.yaml + mas/access/conditions.py) - by the time request_vm actually
runs here, the call has already been cleared (either under quota, or approved by a human).
"""

from __future__ import annotations

from datetime import datetime, timezone

from mcp.server.mcpserver import Context, MCPServer

from mas.mcp.caller import caller_from_ctx
from mas.store.document_store import DocumentStore


def build_server(store: DocumentStore) -> MCPServer:
    server = MCPServer("resources")

    @server.tool(
        description="Check VM quota for a team. Defaults to the caller's own team; only "
        "managers may pass a different team.",
        structured_output=True,
    )
    async def check_quota(team: str | None, ctx: Context) -> dict[str, object]:
        caller = caller_from_ctx(ctx)
        target_team = team or caller.team
        if target_team != caller.team and not caller.has_role("manager"):
            return {"error": "you may only check your own team's quota"}
        quota = await store.get("quotas", target_team)
        if quota is None:
            return {"error": f"no quota record for team '{target_team}'"}
        return quota

    @server.tool(
        description="Request one or more VMs for the caller's team. May require approval "
        "if it would exceed the team's quota (handled upstream).",
        structured_output=True,
    )
    async def request_vm(count: int, purpose: str, ctx: Context) -> dict[str, object]:
        caller = caller_from_ctx(ctx)
        quota = await store.get("quotas", caller.team)
        if quota is None:
            return {"error": f"no quota record for team '{caller.team}'"}
        new_used = quota["vm_used"] + count
        await store.update("quotas", caller.team, {"vm_used": new_used})
        request = {
            "id": store.next_id("vm_requests", "VM"),
            "requester": caller.user_id,
            "team": caller.team,
            "count": count,
            "purpose": purpose,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.insert("vm_requests", request)
        return {
            "request_id": request["id"],
            "count": count,
            "team": caller.team,
            "vm_used": new_used,
            "vm_limit": quota["vm_limit"],
        }

    @server.tool(description="Request a laptop of a given model for the caller.", structured_output=True)
    async def request_laptop(model: str, ctx: Context) -> dict[str, object]:
        caller = caller_from_ctx(ctx)
        item = await store.get("inventory", model)
        if item is None or item["stock"] <= 0:
            return {"error": f"'{model}' is out of stock"}
        await store.update("inventory", model, {"stock": item["stock"] - 1})
        asset = {
            "id": store.next_id("assets", "AST"),
            "type": "laptop",
            "model": model,
            "assignee": caller.user_id,
        }
        await store.insert("assets", asset)
        return asset

    return server
