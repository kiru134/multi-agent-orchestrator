"""Wires the whole system together: store, MCP servers, agent auth, graphs. Used by both
the FastAPI app and the CLI so they build the exact same runtime."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import httpx
from fastapi import FastAPI

from mas.a2a.client_transport import A2AClientTransport, LOOPBACK_BASE_URL, discover_agents
from mas.a2a.server import mount_agent_a2a
from mas.access.policy import PolicyEngine, get_policy_engine
from mas.agents.base import BaseAgent
from mas.agents.registry import load_agent_definitions
from mas.auth.agent_auth import AgentSessionRegistry, agent_secret, run_handshake
from mas.config import settings_config
from mas.graphs.memory import SessionRegistry, ThreadLockRegistry, get_checkpointer
from mas.graphs.orchestrator import build_orchestrator_graph
from mas.graphs.subagent import build_subagent_graph
from mas.llm.availability import detect_available_providers
from mas.llm.factory import get_chat_model_for
from mas.llm.selector import ModelSelector
from mas.mcp.client_manager import MCPClientManager
from mas.orchestrator.transport import AgentTransport, InProcessTransport
from mas.paths import SEED_DIR
from mas.store.document_store import DocumentStore

logger = logging.getLogger("mas.runtime")
logging.getLogger("a2a").setLevel(logging.WARNING)  # the SDK logs every card/request body at INFO


@dataclass
class Runtime:
    store: DocumentStore
    mcp: MCPClientManager
    policy: PolicyEngine
    agent_defs: dict[str, BaseAgent]
    agent_registry: AgentSessionRegistry
    session_registry: SessionRegistry
    thread_locks: ThreadLockRegistry
    transport: AgentTransport
    subagent_graphs: dict[str, object]
    orchestrator_graph: object
    unauthenticated_agents: list[str]
    transport_mode: str
    discovered_agents: dict | None = None
    _a2a_http_client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        if self._a2a_http_client is not None:
            await self._a2a_http_client.aclose()
        await self.mcp.aclose()


async def build_runtime(*, backend: str | None = None, transport_mode: str = "a2a") -> Runtime:
    """`backend`: None = auto-detect from the environment (Anthropic key / local Ollama),
    or force "anthropic" / "ollama" explicitly.

    `transport_mode`: "a2a" (default) - sub-agents each run a real A2A server (Agent Card +
    JSON-RPC) on an internal loopback app, and the orchestrator discovers their capabilities
    by fetching those cards instead of reading config/agents.yaml descriptions, then calls
    them over real A2A. "inprocess" calls each agent's graph directly in Python instead (no
    HTTP/A2A at all) - kept as a simpler fallback/comparison path.
    """
    store = DocumentStore()
    store.seed(SEED_DIR)

    policy = get_policy_engine()
    agent_defs = load_agent_definitions()

    mcp = MCPClientManager()
    await mcp.connect_all(agent_defs, store)

    agent_registry = AgentSessionRegistry()
    unauthenticated: list[str] = []
    for agent_id in agent_defs:
        session = run_handshake(agent_registry, agent_id, agent_secret(agent_id))
        if session is None:
            unauthenticated.append(agent_id)
            logger.warning("agent '%s' failed the auth handshake - excluded from routing", agent_id)

    available = {backend} if backend else await detect_available_providers()
    if not available:
        logger.warning("no LLM provider available (no ANTHROPIC_API_KEY, no Ollama server) - LLM calls will fail")
    selector = ModelSelector(available_providers=available or None)

    mem_cfg = settings_config()["memory"]
    checkpointer = get_checkpointer()

    subagent_graphs = {}
    for agent_id, adef in agent_defs.items():
        choice = selector.for_tier(adef.model_tier, reason=f"agent default ({agent_id})")
        llm = get_chat_model_for(choice)
        subagent_graphs[agent_id] = build_subagent_graph(
            adef, llm, mcp, policy, store, checkpointer, max_turns=mem_cfg["max_turns"]
        )

    # Backstop recursion limits (generous headroom above each agent's real iterations cap -
    # see mas/graphs/subagent.py) in case that in-graph cap is ever bypassed by a bug.
    recursion_limits = {agent_id: 2 * adef.max_iterations + 6 for agent_id, adef in agent_defs.items()}

    orchestrator_agent_defs = agent_defs  # may be replaced below with discovery-sourced descriptions
    a2a_http_client: httpx.AsyncClient | None = None
    discovered: dict | None = None

    if transport_mode == "a2a":
        a2a_internal_app = FastAPI()
        for agent_id, adef in agent_defs.items():
            mount_agent_a2a(
                a2a_internal_app, adef, subagent_graphs[agent_id], mcp, LOOPBACK_BASE_URL,
                recursion_limit=recursion_limits[agent_id],
            )
        a2a_http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=a2a_internal_app), base_url=LOOPBACK_BASE_URL
        )
        discovered = await discover_agents(a2a_internal_app, list(agent_defs.keys()), a2a_http_client)
        transport: AgentTransport = A2AClientTransport(a2a_http_client, discovered)
        # The orchestrator's view of "what can this agent do" now comes from the live
        # Agent Card it fetched, not agent.description directly - a shallow copy per agent
        # so this never mutates the shared agent instance used everywhere else.
        orchestrator_agent_defs = {}
        for agent_id, adef in agent_defs.items():
            view = copy.copy(adef)
            view.description = discovered[agent_id].description
            orchestrator_agent_defs[agent_id] = view
    else:
        transport = InProcessTransport(subagent_graphs, recursion_limits=recursion_limits)

    orch_choice = selector.select("orchestrator.main")
    orch_llm = get_chat_model_for(orch_choice)
    orchestrator_graph = build_orchestrator_graph(
        orchestrator_agent_defs,
        agent_registry,
        policy,
        transport,
        orch_llm,
        checkpointer,
        max_turns=mem_cfg["max_turns"],
        max_delegation_rounds=mem_cfg["max_iterations_default"],
    )

    return Runtime(
        store=store,
        mcp=mcp,
        policy=policy,
        agent_defs=agent_defs,
        agent_registry=agent_registry,
        session_registry=SessionRegistry(),
        thread_locks=ThreadLockRegistry(),
        transport=transport,
        subagent_graphs=subagent_graphs,
        orchestrator_graph=orchestrator_graph,
        unauthenticated_agents=unauthenticated,
        transport_mode=transport_mode,
        discovered_agents=discovered,
        _a2a_http_client=a2a_http_client,
    )
