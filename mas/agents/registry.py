"""Loads every registered agent and cross-checks it against config/agents.yaml's
ops/security entries (secret, tool risk/approval policy) - every agent needs one, and
catching a missing entry here beats discovering it mid-conversation."""

from __future__ import annotations

# Importing these registers each agent class with AgentRegistry (see mas/agents/base.py).
# Adding a new agent means adding one such module and importing it here - nothing else in
# this file changes.
import mas.agents.infra  # noqa: F401
import mas.agents.it_support  # noqa: F401
import mas.agents.resource_alloc  # noqa: F401
from mas.agents.base import AgentRegistry, BaseAgent
from mas.config import agents_config


def load_agent_definitions() -> dict[str, BaseAgent]:
    agents = AgentRegistry.all()
    ops_cfg = agents_config().get("agents", {})
    missing = [agent_id for agent_id in agents if agent_id not in ops_cfg]
    if missing:
        raise KeyError(
            f"agent(s) {missing} are registered as classes but have no ops entry "
            f"(secret_env, tools policy) in config/agents.yaml"
        )
    return agents
