"""Adapts an existing compiled sub-agent LangGraph graph to the A2A `AgentExecutor`
interface. The graph itself doesn't change at all - this is purely a protocol adapter, same
spirit as `InProcessTransport` but speaking A2A instead of being called directly in Python.
"""

from __future__ import annotations

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils.message import new_agent_text_message
from langchain_core.messages import AIMessage, HumanMessage

from mas.graphs.memory import thread_config


class SubagentA2AExecutor(AgentExecutor):
    def __init__(self, agent_id: str, graph, recursion_limit: int = 50) -> None:
        self._agent_id = agent_id
        self._graph = graph
        self._recursion_limit = recursion_limit

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_text = context.get_user_input()
        meta = context.metadata or {}
        caller_meta = meta.get("caller")
        user_id = meta.get("user_id", "unknown")
        session_id = meta.get("session_id", "default")

        if not caller_meta:
            await event_queue.enqueue_event(
                new_agent_text_message(
                    "ERROR: no caller identity was provided with this request - refusing.",
                    context_id=context.context_id,
                    task_id=context.task_id,
                )
            )
            return

        cfg = {**thread_config(user_id, session_id, self._agent_id), "recursion_limit": self._recursion_limit}
        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=task_text)], "caller": caller_meta, "iterations": 0},
            config=cfg,
        )
        messages = result.get("messages", [])
        last_answer = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None
        )
        answer = last_answer.content if last_answer else "(no answer produced)"
        await event_queue.enqueue_event(
            new_agent_text_message(answer, context_id=context.context_id, task_id=context.task_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Not supported in this prototype - each call runs to completion (or its own
        # recursion-limit backstop) before a response is ever enqueued.
        raise NotImplementedError("cancellation is not supported by this agent")
