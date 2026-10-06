from __future__ import annotations

from mas.agents.base import AgentRegistry, BaseAgent


@AgentRegistry.register
class ResourceAllocAgent(BaseAgent):
    agent_id = "resource_alloc"
    description = (
        'Resource allocation agent for REQUESTING new resources: provisioning a new '
        'VM/server, requesting a new laptop, or checking a team\'s VM quota. Use this - not '
        'infra - for any request containing phrases like "I need a VM", "request a '
        'VM/server", "new laptop", or "provision".'
    )
    mcp_server_name = "resources"
    mcp_server_factory = "mas.mcp.mock_servers.resources_server:build_server"
    system_prompt_file = "config/prompts/resource_alloc.md"
