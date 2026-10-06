from __future__ import annotations

from mas.agents.base import AgentRegistry, BaseAgent


@AgentRegistry.register
class ITSupportAgent(BaseAgent):
    agent_id = "it_support"
    description = (
        "IT support agent for problems and requests from a specific user: filing a support "
        "ticket (broken/malfunctioning laptop, software issues, access problems), checking "
        "ticket status, or resetting a password. Does NOT provision new VMs or hardware - "
        "use resource_alloc for that."
    )
    mcp_server_name = "it"
    mcp_server_factory = "mas.mcp.mock_servers.it_server:build_server"
    system_prompt_file = "config/prompts/it_support.md"
