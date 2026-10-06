"""Mock Infra MCP server: service status, incidents, restarts, scaling.

Every tool reads/writes the shared DocumentStore passed in at construction time - there is
no state inside this module. `build_server(store)` is referenced by name from
config/agents.yaml (`factory: mas.mcp.mock_servers.infra_server:build_server`).
"""

from __future__ import annotations

from datetime import datetime, timezone

from mcp.server.mcpserver import Context, MCPServer

from mas.store.document_store import DocumentStore


def build_server(store: DocumentStore) -> MCPServer:
    server = MCPServer("infra")

    @server.tool(description="Get the current status of a service by name.", structured_output=True)
    async def get_service_status(service: str, ctx: Context) -> dict[str, object]:
        svc = await store.get("services", service)
        if svc is None:
            return {"error": f"no such service '{service}'"}
        return svc

    @server.tool(
        description="List incidents, optionally filtered to one service. Unresolved first.",
        structured_output=True,
    )
    async def list_incidents(service: str | None = None) -> dict[str, object]:
        def match(doc: dict) -> bool:
            return service is None or doc["service_id"] == service

        incidents = await store.find("incidents", match)
        incidents.sort(key=lambda d: (d["resolved"], d["created_at"]))
        return {"incidents": incidents}

    @server.tool(
        description="Restart a service. Brings a degraded/unhealthy service back to healthy.",
        structured_output=True,
    )
    async def restart_service(service: str, ctx: Context) -> dict[str, object]:
        svc = await store.get("services", service)
        if svc is None:
            return {"error": f"no such service '{service}'"}
        before = svc["status"]
        updated = await store.update("services", service, {"status": "healthy"})
        await store.insert(
            "incidents",
            {
                "id": store.next_id("incidents", "INC"),
                "service_id": service,
                "message": f"Manual restart ({before} -> healthy)",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "resolved": True,
            },
        )
        return {"service": service, "status_before": before, "status_after": updated["status"]}

    @server.tool(
        description="Scale a deployment to a target replica count.",
        structured_output=True,
    )
    async def scale_deployment(service: str, replicas: int, ctx: Context) -> dict[str, object]:
        svc = await store.get("services", service)
        if svc is None:
            return {"error": f"no such service '{service}'"}
        if replicas < 1:
            return {"error": "replicas must be at least 1"}
        before = svc["replicas"]
        updated = await store.update("services", service, {"replicas": replicas})
        return {"service": service, "replicas_before": before, "replicas_after": updated["replicas"]}

    return server
