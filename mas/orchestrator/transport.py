"""How the orchestrator actually invokes a sub-agent. `InProcessTransport` runs the
sub-agent's compiled graph in this same process; a later HTTP/MCP-based transport could
implement the same interface without the orchestrator graph changing at all.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from mas.access.models import Caller
from mas.graphs.memory import thread_config
from mas.orchestrator.events import StreamEvent


class AgentTransport(Protocol):
    def has_agent(self, agent_id: str) -> bool: ...

    def invoke_stream(
        self, agent_id: str, task: str, caller: Caller, user_id: str, session_id: str
    ) -> AsyncIterator[StreamEvent]: ...


class InProcessTransport:
    def __init__(self, graphs: dict, recursion_limits: dict[str, int] | None = None, default_recursion_limit: int = 50) -> None:
        self._graphs = graphs
        # The real cap on tool-call rounds is enforced inside each sub-agent graph itself
        # (AgentState.iterations vs. agent_def.max_iterations - see mas/graphs/subagent.py),
        # which lets the model wrap up gracefully. This recursion_limit is just a generous
        # backstop against runaway loops (e.g. a bug) hitting LangGraph's own hard ceiling,
        # which raises GraphRecursionError instead of letting the model respond at all.
        self._recursion_limits = recursion_limits or {}
        self._default_recursion_limit = default_recursion_limit

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self._graphs

    def graph_for(self, agent_id: str):
        return self._graphs.get(agent_id)

    async def invoke_stream(
        self, agent_id: str, task: str, caller: Caller, user_id: str, session_id: str
    ) -> AsyncIterator[StreamEvent]:
        graph = self._graphs[agent_id]
        limit = self._recursion_limits.get(agent_id, self._default_recursion_limit)
        cfg = {**thread_config(user_id, session_id, agent_id), "recursion_limit": limit}

        yield StreamEvent("agent_start", agent_id, {"task": task})
        try:
            async for mode, payload in graph.astream(
                {"messages": [HumanMessage(content=task)], "caller": caller.to_meta(), "iterations": 0},
                config=cfg,
                stream_mode=["messages", "custom"],
            ):
                if mode == "messages":
                    chunk, _meta = payload
                    # AIMessageChunk (real streaming providers) and plain AIMessage (the
                    # fake model, which has no true token streaming) both subclass AIMessage.
                    if isinstance(chunk, AIMessage) and chunk.content:
                        text = (
                            chunk.content
                            if isinstance(chunk.content, str)
                            else "".join(b.get("text", "") for b in chunk.content if isinstance(b, dict))
                        )
                        if text:
                            yield StreamEvent("token", agent_id, {"text": text})
                elif mode == "custom":
                    kind = payload.get("type", "info")
                    source = payload.get("agent", agent_id)
                    rest = {k: v for k, v in payload.items() if k not in ("type", "agent")}
                    yield StreamEvent(kind, source, rest)
        except GraphRecursionError:
            yield StreamEvent("error", agent_id, {"message": "stopped after too many tool-call iterations"})

        state = await graph.aget_state(cfg)
        messages = state.values.get("messages", [])
        last_answer = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None
        )
        final_text = last_answer.content if last_answer else "(no answer produced)"
        yield StreamEvent("agent_end", agent_id, {"answer": final_text})
