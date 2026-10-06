from langchain_core.messages import AIMessage, HumanMessage

from mas.access.models import Caller
from mas.auth.agent_auth import AgentSessionRegistry, agent_secret, run_handshake
from mas.config import agents_config
from mas.graphs.memory import thread_config
from mas.graphs.orchestrator import build_orchestrator_graph
from mas.graphs.subagent import build_subagent_graph
from mas.llm.fake import FakeToolChatModel
from mas.orchestrator.transport import InProcessTransport

ALICE = Caller(user_id="alice", team="platform", roles=("engineer",))
CAROL = Caller(user_id="carol", team="sales", roles=("employee",))


def _authenticated_registry() -> AgentSessionRegistry:
    registry = AgentSessionRegistry()
    for agent_id in agents_config()["agents"]:
        assert run_handshake(registry, agent_id, agent_secret(agent_id)) is not None
    return registry


async def test_allowed_delegation_end_to_end(agent_defs, mcp, policy, checkpointer, store):
    registry = _authenticated_registry()
    infra_llm = FakeToolChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "infra.get_service_status", "args": {"service": "payments"}, "id": "c1"}]),
            AIMessage(content="Payments is currently degraded."),
        ]
    )
    transport = InProcessTransport({"infra": build_subagent_graph(agent_defs["infra"], infra_llm, mcp, policy, store, checkpointer, 16)})
    orch_llm = FakeToolChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "delegate_to_infra", "args": {"task": "check payments"}, "id": "o1"}]),
            AIMessage(content="Payments is degraded - infra is aware."),
        ]
    )
    graph = build_orchestrator_graph(agent_defs, registry, policy, transport, orch_llm, checkpointer, max_turns=16)
    cfg = thread_config("alice", "s1", "orchestrator")
    cfg["recursion_limit"] = 25
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="status?")], "caller": ALICE.to_meta(), "user_id": "alice", "session_id": "s1"},
        config=cfg,
    )
    assert result["messages"][-1].content == "Payments is degraded - infra is aware."


async def test_agent_level_denial_never_invokes_subagent(agent_defs, mcp, policy, checkpointer, store):
    registry = _authenticated_registry()
    infra_llm = FakeToolChatModel(responses=[AIMessage(content="should never run")])
    infra_graph = build_subagent_graph(agent_defs["infra"], infra_llm, mcp, policy, store, checkpointer, 16)
    transport = InProcessTransport({"infra": infra_graph})
    orch_llm = FakeToolChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "delegate_to_infra", "args": {"task": "restart payments"}, "id": "o1"}]),
            AIMessage(content="You don't have access to infra."),
        ]
    )
    graph = build_orchestrator_graph(agent_defs, registry, policy, transport, orch_llm, checkpointer, max_turns=16)
    cfg = thread_config("carol", "s1", "orchestrator")
    cfg["recursion_limit"] = 25
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="restart payments")], "caller": CAROL.to_meta(), "user_id": "carol", "session_id": "s1"},
        config=cfg,
    )
    tool_msgs = [m for m in result["messages"] if hasattr(m, "tool_call_id")]
    assert len(tool_msgs) == 1 and tool_msgs[0].content.startswith("DENIED")
    svc = await store.get("services", "payments")
    assert svc["status"] == "degraded"  # infra sub-agent never ran, store untouched
    infra_state = await infra_graph.aget_state(thread_config("carol", "s1", "infra"))
    assert infra_state.values == {} or not infra_state.values.get("messages")  # no thread was ever started


async def test_unauthenticated_agent_rejected_even_if_llm_calls_it(agent_defs, mcp, policy, checkpointer, store):
    # Simulate infra presenting the wrong shared secret at handshake time (e.g. a
    # misconfigured agent process) - its startup handshake fails and it must stay excluded.
    registry = AgentSessionRegistry()
    for agent_id in agents_config()["agents"]:
        claimed_secret = "not-the-real-secret" if agent_id == "infra" else agent_secret(agent_id)
        run_handshake(registry, agent_id, claimed_secret)
    assert not registry.is_authenticated("infra")

    infra_llm = FakeToolChatModel(responses=[AIMessage(content="should never run")])
    transport = InProcessTransport({"infra": build_subagent_graph(agent_defs["infra"], infra_llm, mcp, policy, store, checkpointer, 16)})
    # The fake LLM calls delegate_to_infra anyway (as if it had hallucinated a tool it was
    # never actually given) - delegate_node must still refuse it.
    orch_llm = FakeToolChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "delegate_to_infra", "args": {"task": "check status"}, "id": "o1"}]),
            AIMessage(content="infra is currently unavailable."),
        ]
    )
    graph = build_orchestrator_graph(agent_defs, registry, policy, transport, orch_llm, checkpointer, max_turns=16)
    cfg = thread_config("alice", "s3", "orchestrator")
    cfg["recursion_limit"] = 25
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="status?")], "caller": ALICE.to_meta(), "user_id": "alice", "session_id": "s3"},
        config=cfg,
    )
    tool_msgs = [m for m in result["messages"] if hasattr(m, "tool_call_id")]
    assert len(tool_msgs) == 1
    assert "not authenticated" in tool_msgs[0].content.lower() or tool_msgs[0].content.startswith("ERROR")


async def test_threads_are_isolated_per_agent(agent_defs, mcp, policy, checkpointer, store):
    registry = _authenticated_registry()
    infra_llm = FakeToolChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "infra.get_service_status", "args": {"service": "payments"}, "id": "c1"}]),
            AIMessage(content="Payments is degraded."),
        ]
    )
    infra_graph = build_subagent_graph(agent_defs["infra"], infra_llm, mcp, policy, store, checkpointer, 16)
    transport = InProcessTransport({"infra": infra_graph})
    orch_llm = FakeToolChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "delegate_to_infra", "args": {"task": "check payments status"}, "id": "o1"}]),
            AIMessage(content="Payments is degraded."),
        ]
    )
    graph = build_orchestrator_graph(agent_defs, registry, policy, transport, orch_llm, checkpointer, max_turns=16)
    cfg = thread_config("alice", "s4", "orchestrator")
    cfg["recursion_limit"] = 25
    await graph.ainvoke(
        {"messages": [HumanMessage(content="status?")], "caller": ALICE.to_meta(), "user_id": "alice", "session_id": "s4"},
        config=cfg,
    )

    orch_state = await graph.aget_state(cfg)
    infra_state = await infra_graph.aget_state(thread_config("alice", "s4", "infra"))

    orch_texts = [m.content for m in orch_state.values["messages"] if isinstance(m, HumanMessage)]
    infra_texts = [m.content for m in infra_state.values["messages"] if isinstance(m, HumanMessage)]
    assert orch_texts == ["status?"]
    assert infra_texts == ["check payments status"]
    # the orchestrator's own thread never sees the infra agent's internal tool call/result
    assert not any(hasattr(m, "tool_call_id") and m.name == "infra.get_service_status" for m in orch_state.values["messages"])
