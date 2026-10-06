"""Connects to each agent's MCP server and exposes its tools to the graphs.

Each `BaseAgent` names its own server via `mcp_server_factory` ("module:callable") and
`mcp_transport` (see mas/agents/base.py) - this is a code/identity concern, not config, so
it comes from the agent object, not config/agents.yaml. For this prototype every server is
"inprocess": we connect to it with mcp's in-memory transport (`mcp.Client(server_instance)`).
A `stdio` or `streamable_http` transport is a drop-in swap for a real server later - only
`connect_all` would change, nothing downstream.

Caller identity travels in the MCP request's `_meta` (`{"caller": {...}}`), set on every
`call_tool`, never on the tool schema the LLM sees - the LLM cannot forge it.
"""

from __future__ import annotations

import importlib
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from mcp import Client, types as mcp_types
from mcp.server.mcpserver import MCPServer

from mas.access.models import Caller
from mas.store.document_store import DocumentStore

if TYPE_CHECKING:
    from mas.agents.base import BaseAgent

ServerFactory = Callable[[DocumentStore], MCPServer]


@dataclass(frozen=True)
class MCPToolSpec:
    """A tool, namespaced as "<server>.<tool>" - this namespaced form is what appears in
    config/policies.yaml and config/agents.yaml, and what the LLM sees as the tool name."""

    namespaced_name: str
    server: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]


def _resolve_factory(path: str) -> ServerFactory:
    module_name, _, func_name = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


class MCPClientManager:
    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._clients: dict[str, Client] = {}
        self._tools: dict[str, list[MCPToolSpec]] = {}
        self._connected = False

    async def connect_all(self, agents: dict[str, "BaseAgent"], store: DocumentStore) -> None:
        for agent in agents.values():
            for name in agent.mcp_server_names:
                if name in self._clients:
                    continue  # already connected (servers may be shared, though not today)
                if agent.mcp_transport != "inprocess":
                    raise NotImplementedError(
                        f"transport '{agent.mcp_transport}' not implemented in this "
                        f"prototype (server '{name}') - only 'inprocess' is wired up"
                    )
                factory = _resolve_factory(agent.mcp_server_factory)
                server = factory(store)
                await self._connect_one(name, server)
        self._connected = True

    async def _connect_one(self, name: str, server: MCPServer) -> None:
        client = Client(server)
        connected = await self._exit_stack.enter_async_context(client)
        self._clients[name] = connected
        result = await connected.list_tools()
        self._tools[name] = [
            MCPToolSpec(
                namespaced_name=f"{name}.{t.name}",
                server=name,
                tool_name=t.name,
                description=t.description or "",
                input_schema=t.input_schema,
            )
            for t in result.tools
        ]

    def tools_for_server(self, server: str) -> list[MCPToolSpec]:
        return list(self._tools.get(server, []))

    def tools_for_servers(self, servers: list[str]) -> list[MCPToolSpec]:
        out: list[MCPToolSpec] = []
        for s in servers:
            out.extend(self.tools_for_server(s))
        return out

    def tool_count(self) -> int:
        return sum(len(v) for v in self._tools.values())

    async def call_tool(self, namespaced_name: str, args: dict[str, Any], caller: Caller) -> mcp_types.CallToolResult:
        server, _, tool_name = namespaced_name.partition(".")
        client = self._clients.get(server)
        if client is None:
            raise KeyError(f"no connected MCP server named '{server}'")
        return await client.call_tool(tool_name, args, meta={"caller": caller.to_meta()})

    async def aclose(self) -> None:
        await self._exit_stack.aclose()
        self._clients.clear()
        self._connected = False
