"""The extension point for sub-agents.

To add a new agent: write one file (see mas/agents/infra.py for the simplest possible
example), subclass `BaseAgent`, set the four required class attributes, decorate it with
`@AgentRegistry.register`, and import that module once from mas/agents/registry.py. Nothing
else in the codebase changes (Open/Closed) - the generic sub-agent graph, the MCP client
manager, the A2A server/card builders, and the orchestrator all consume `BaseAgent`
uniformly through this base class (Dependency Inversion), never a concrete agent type.

Deliberately NOT here: secrets and approval/risk policy (config/agents.yaml, consumed by
mas/auth/agent_auth.py and mas/access/approval.py). Those are security/ops configuration
that a security team should be able to change without touching Python, so they stay
data-driven even though an agent's identity and behavior are now real classes.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from mas.config import prompt_text


class BaseAgent(ABC):
    """Base class for one sub-agent's identity and behavior.

    The four ClassVars below are all a simple agent needs to provide. Override a method
    only when an agent needs behavior the generic flow doesn't already give it - e.g. a
    dynamically-built prompt, or (via `build_graph`) an entirely different state machine
    instead of the standard trim -> llm -> tools loop every other agent uses.
    """

    agent_id: ClassVar[str]
    description: ClassVar[str]
    mcp_server_factory: ClassVar[str]  # "module:callable" - see MCPClientManager
    system_prompt_file: ClassVar[str]
    default_model_tier: ClassVar[str] = "standard"
    max_iterations: ClassVar[int] = 6
    mcp_transport: ClassVar[str] = "inprocess"  # "stdio"/"streamable_http" for a real server later
    # Defaults to agent_id; override when the MCP server's (and so every tool's namespace
    # prefix, e.g. "it.create_ticket") name differs from the agent's own id.
    mcp_server_name: ClassVar[str | None] = None

    def __init__(self) -> None:
        if not getattr(self, "agent_id", None):
            raise TypeError(f"{type(self).__name__} must set agent_id")

    # ---- identity, as read by the generic graph/A2A/MCP-connection code ----------------

    @property
    def mcp_server_names(self) -> list[str]:
        """One server per agent in this prototype - a subclass needing more than one
        overrides this instead of mcp_server_factory."""
        return [self.mcp_server_name or self.agent_id]

    @property
    def model_tier(self) -> str:
        return self.default_model_tier

    @property
    def system_prompt(self) -> str:
        """Override for a prompt computed at runtime instead of a static file."""
        return prompt_text(self.system_prompt_file)

    # ---- behavior: overridable for an agent that needs something non-generic ----------

    def build_graph(self, llm, mcp, policy, store, checkpointer, max_turns):
        """Compiles this agent's LangGraph state machine. The default is the shared
        trim -> llm -> tools loop (mas/graphs/subagent.py); override for an agent whose
        control flow genuinely needs to differ, not just its prompt or tools."""
        from mas.graphs.subagent import build_subagent_graph

        return build_subagent_graph(self, llm, mcp, policy, store, checkpointer, max_turns)


class AgentRegistry:
    """Decouples "which agents exist" from "who builds/consumes them" (Registry pattern) -
    a new agent self-registers by being decorated, and every consumer (MCPClientManager,
    the orchestrator, the A2A mounting code) just asks the registry for the current set."""

    _classes: dict[str, type[BaseAgent]] = {}

    @classmethod
    def register(cls, agent_cls: type[BaseAgent]) -> type[BaseAgent]:
        if not getattr(agent_cls, "agent_id", None):
            raise TypeError(f"{agent_cls.__name__} must set agent_id before registering")
        cls._classes[agent_cls.agent_id] = agent_cls
        return agent_cls

    @classmethod
    def all(cls) -> dict[str, BaseAgent]:
        return {agent_id: agent_cls() for agent_id, agent_cls in cls._classes.items()}
