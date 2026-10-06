"""Builds each agent's real A2A AgentCard - one MCP tool becomes one AgentSkill. This is
what the orchestrator now discovers instead of reading `description` straight out of
config/agents.yaml (see mas/a2a/client_transport.py)."""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from mas.agents.base import BaseAgent
from mas.mcp.client_manager import MCPClientManager


def rpc_url(base_url: str, agent_id: str) -> str:
    return f"{base_url.rstrip('/')}/agents/{agent_id}/a2a"


def agent_card_path(agent_id: str) -> str:
    return f"/agents/{agent_id}/.well-known/agent-card.json"


def build_agent_card(agent_def: BaseAgent, mcp: MCPClientManager, base_url: str) -> AgentCard:
    specs = mcp.tools_for_servers(agent_def.mcp_server_names)
    skills = [
        AgentSkill(
            id=spec.namespaced_name,
            name=spec.tool_name,
            description=spec.description or spec.namespaced_name,
            tags=[agent_def.agent_id],
        )
        for spec in specs
    ]
    return AgentCard(
        name=agent_def.agent_id,
        description=agent_def.description,
        url=rpc_url(base_url, agent_def.agent_id),
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=skills,
    )
