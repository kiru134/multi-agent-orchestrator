"""The Infra agent. This is the minimal case: nothing but identity - the generic graph,
tool-filtering, and A2A machinery handle everything else."""

from __future__ import annotations

from mas.agents.base import AgentRegistry, BaseAgent


@AgentRegistry.register
class InfraAgent(BaseAgent):
    agent_id = "infra"
    description = (
        "Infra agent for EXISTING deployed services only (payments, auth, search): checking "
        "status, listing incidents, restarting, or scaling a deployment that already exists. "
        "Does NOT provision new VMs, servers, or hardware - use resource_alloc for any "
        "request to get/request a new VM or device."
    )
    mcp_server_factory = "mas.mcp.mock_servers.infra_server:build_server"
    system_prompt_file = "config/prompts/infra.md"
