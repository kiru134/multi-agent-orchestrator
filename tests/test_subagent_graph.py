from langchain_core.messages import AIMessage, HumanMessage

from mas.access.models import Caller
from mas.graphs.memory import thread_config
from mas.graphs.subagent import build_subagent_graph
from mas.llm.fake import FakeToolChatModel

ALICE = Caller(user_id="alice", team="platform", roles=("engineer",))
CAROL = Caller(user_id="carol", team="sales", roles=("employee",))


async def test_subagent_calls_tool_and_answers(agent_defs, mcp, policy, store, checkpointer):
    llm = FakeToolChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "infra.get_service_status", "args": {"service": "payments"}, "id": "c1"}]),
            AIMessage(content="Payments is degraded."),
        ]
    )
    graph = build_subagent_graph(agent_defs["infra"], llm, mcp, policy, store, checkpointer, max_turns=16)
    cfg = thread_config("alice", "s1", "infra")
    cfg["recursion_limit"] = 10
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="check payments")], "caller": ALICE.to_meta()}, config=cfg
    )
    final = result["messages"][-1]
    assert final.content == "Payments is degraded."
    assert any(m.content.startswith('{"id": "payments"') for m in result["messages"] if hasattr(m, "tool_call_id"))


async def test_subagent_denies_tool_call_defense_in_depth(agent_defs, mcp, policy, checkpointer, store):
    # carol has no role granting any infra.* tool. Even if something upstream mistakenly
    # let her reach the infra sub-agent, authorize_tools must still block the tool call.
    llm = FakeToolChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "infra.restart_service", "args": {"service": "payments"}, "id": "c1"}]),
            AIMessage(content="I could not do that - you lack permission."),
        ]
    )
    graph = build_subagent_graph(agent_defs["infra"], llm, mcp, policy, store, checkpointer, max_turns=16)
    cfg = thread_config("carol", "s1", "infra")
    cfg["recursion_limit"] = 10
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="restart payments")], "caller": CAROL.to_meta()}, config=cfg
    )
    tool_messages = [m for m in result["messages"] if hasattr(m, "tool_call_id")]
    assert len(tool_messages) == 1
    assert tool_messages[0].content.startswith("DENIED")
    svc = await store.get("services", "payments")
    assert svc["status"] == "degraded"  # unchanged - the tool call never reached MCP


async def test_subagent_llm_only_offered_allowed_tools(agent_defs, mcp, policy, store, checkpointer):
    """Indirect check: an engineer asking the infra agent to restart should get a plain
    text answer declining (no tool_calls at all), since restart_service isn't bound for
    them - the script only has one response, proving the node never needed a second LLM call
    through a tool loop because no tool was offered/called."""
    llm = FakeToolChatModel(responses=[AIMessage(content="I can't restart services - only scale or check status.")])
    graph = build_subagent_graph(agent_defs["infra"], llm, mcp, policy, store, checkpointer, max_turns=16)
    cfg = thread_config("alice", "s2", "infra")
    cfg["recursion_limit"] = 10
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="restart payments")], "caller": ALICE.to_meta()}, config=cfg
    )
    assert len(result["messages"]) == 2  # human + the single plain AI reply, no tool round-trip
