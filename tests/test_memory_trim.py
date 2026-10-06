from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from mas.graphs.memory import thread_id, trim_history


def _msg(cls, **kw):
    m = cls(**kw)
    return m


def test_no_trim_needed_under_budget():
    msgs = [HumanMessage(content="hi", id="1"), AIMessage(content="hello", id="2")]
    assert trim_history(msgs, max_turns=5) == []


def test_trims_whole_turns_from_the_front():
    msgs = [
        HumanMessage(content="turn1", id="h1"),
        AIMessage(content="r1", id="a1"),
        HumanMessage(content="turn2", id="h2"),
        AIMessage(content="r2", id="a2"),
        HumanMessage(content="turn3", id="h3"),
        AIMessage(content="r3", id="a3"),
    ]
    removed = trim_history(msgs, max_turns=2)
    removed_ids = {r.id for r in removed}
    assert removed_ids == {"h1", "a1"}  # only the oldest turn is dropped


def test_never_splits_a_tool_call_from_its_result():
    msgs = [
        HumanMessage(content="turn1", id="h1"),
        AIMessage(content="", id="a1", tool_calls=[{"name": "x", "args": {}, "id": "c1"}]),
        ToolMessage(content="result", id="t1", tool_call_id="c1", name="x"),
        AIMessage(content="done", id="a1b"),
        HumanMessage(content="turn2", id="h2"),
        AIMessage(content="r2", id="a2"),
    ]
    removed = trim_history(msgs, max_turns=1)
    removed_ids = {r.id for r in removed}
    # turn1's AI/tool-call/tool-result/final-answer are removed as a whole unit, never split
    assert removed_ids == {"h1", "a1", "t1", "a1b"}


def test_thread_id_scheme():
    assert thread_id("alice", "s1", "infra") == "alice:s1:infra"
    assert thread_id("alice", "s1", "orchestrator") != thread_id("bob", "s1", "orchestrator")
