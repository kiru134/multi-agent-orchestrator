"""Mounts one A2A server (Agent Card + JSON-RPC endpoint) per sub-agent onto the shared
FastAPI app, each at its own path prefix - genuinely separate A2A agents, one process."""

from __future__ import annotations

from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from fastapi import FastAPI

from mas.a2a.agent_card import agent_card_path, build_agent_card
from mas.a2a.executor import SubagentA2AExecutor
from mas.agents.base import BaseAgent
from mas.mcp.client_manager import MCPClientManager


def mount_agent_a2a(
    app: FastAPI,
    agent_def: BaseAgent,
    graph,
    mcp: MCPClientManager,
    base_url: str,
    recursion_limit: int = 50,
) -> None:
    card = build_agent_card(agent_def, mcp, base_url)
    handler = DefaultRequestHandler(
        agent_executor=SubagentA2AExecutor(agent_def.agent_id, graph, recursion_limit=recursion_limit),
        task_store=InMemoryTaskStore(),
    )
    a2a_app = A2AFastAPIApplication(agent_card=card, http_handler=handler)
    a2a_app.add_routes_to_app(
        app,
        agent_card_url=agent_card_path(agent_def.agent_id),
        rpc_url=f"/agents/{agent_def.agent_id}/a2a",
    )
