"""Mock IT Support MCP server: tickets and password resets, scoped to the caller."""

from __future__ import annotations

from datetime import datetime, timezone

from mcp.server.mcpserver import Context, MCPServer

from mas.mcp.caller import caller_from_ctx
from mas.store.document_store import DocumentStore


def build_server(store: DocumentStore) -> MCPServer:
    server = MCPServer("it")

    @server.tool(description="Create a support ticket for the caller.", structured_output=True)
    async def create_ticket(title: str, description: str, ctx: Context) -> dict[str, object]:
        caller = caller_from_ctx(ctx)
        ticket = {
            "id": store.next_id("tickets", "TKT"),
            "requester": caller.user_id,
            "title": title,
            "description": description,
            "status": "open",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.insert("tickets", ticket)
        return ticket

    @server.tool(description="Get the status of one ticket by id (must be the caller's own).", structured_output=True)
    async def get_ticket_status(ticket_id: str, ctx: Context) -> dict[str, object]:
        caller = caller_from_ctx(ctx)
        ticket = await store.get("tickets", ticket_id)
        if ticket is None:
            return {"error": f"no such ticket '{ticket_id}'"}
        if ticket["requester"] != caller.user_id and not caller.has_role("manager", "it_admin"):
            return {"error": "that ticket does not belong to you"}
        return ticket

    @server.tool(description="List the caller's own open and recent tickets.", structured_output=True)
    async def list_my_tickets(ctx: Context) -> dict[str, object]:
        caller = caller_from_ctx(ctx)
        tickets = await store.find("tickets", lambda t: t["requester"] == caller.user_id)
        tickets.sort(key=lambda t: t["created_at"], reverse=True)
        return {"tickets": tickets}

    @server.tool(description="Reset the password for a given account/email.", structured_output=True)
    async def reset_password(account: str, ctx: Context) -> dict[str, object]:
        caller = caller_from_ctx(ctx)
        record = {
            "id": store.next_id("tickets", "TKT"),
            "requester": caller.user_id,
            "title": f"Password reset: {account}",
            "description": f"Password reset performed for account '{account}' by request of {caller.user_id}.",
            "status": "resolved",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.insert("tickets", record)
        return {"account": account, "status": "reset", "audit_ticket": record["id"]}

    return server
