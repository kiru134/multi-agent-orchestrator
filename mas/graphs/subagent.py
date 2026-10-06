"""Builds one compiled LangGraph StateGraph per sub-agent - a minimal state machine:

    START -> trim_memory -> agent_llm --(tool_calls)--> execute_tools --> agent_llm
                                      \\--(no tool_calls)--> END

Each agent gets its own graph instance (own closures over its MCPClientManager server
names, system prompt, and model), compiled with the shared checkpointer. Combined with the
per-agent thread-id scheme in mas/graphs/memory.py, this is what gives each agent an
isolated conversation history.

Tool access is filtered twice: the LLM is only ever shown (via bind_tools) the tools the
caller is allowed to use, and execute_tools re-checks before every call (defense in depth -
the first filter is a prompt-shaping convenience, not the security boundary).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from mas.access.approval import needs_approval
from mas.access.models import Caller
from mas.access.policy import PolicyEngine
from mas.agents.base import BaseAgent
from mas.graphs.memory import make_trim_node
from mas.mcp.client_manager import MCPClientManager
from mas.store.document_store import DocumentStore


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    caller: dict[str, Any]
    # Tool-call rounds used *this task* - not a reducer, so every new invocation must pass
    # iterations=0 explicitly (see InProcessTransport.invoke_stream) or it would otherwise
    # keep whatever the checkpointer persisted from a previous, unrelated delegation.
    iterations: int


MAX_ITERATIONS_REMINDER = (
    "You have used all {max_iterations} tool-call rounds allowed for this task. Do not call "
    "any more tools. Answer now with your best conclusion from what you've already learned - "
    "say plainly if you couldn't fully complete the task, rather than guessing at a result."
)


def _tool_result_text(result: Any) -> str:
    if result.structured_content is not None:
        return json.dumps(result.structured_content)
    parts = [getattr(c, "text", "") for c in result.content]
    return "\n".join(p for p in parts if p) or json.dumps({"ok": not result.is_error})


def build_subagent_graph(
    agent_def: BaseAgent,
    llm: BaseChatModel,
    mcp: MCPClientManager,
    policy: PolicyEngine,
    store: DocumentStore,
    checkpointer: InMemorySaver,
    max_turns: int,
):
    async def agent_llm_node(state: AgentState) -> dict:
        caller = Caller.from_meta(state["caller"])
        iterations = state.get("iterations", 0)
        out_of_budget = iterations >= agent_def.max_iterations

        system_text = agent_def.system_prompt
        if out_of_budget:
            bound = llm  # no tools bound at all - the model physically cannot call another one
            system_text += "\n\n" + MAX_ITERATIONS_REMINDER.format(max_iterations=agent_def.max_iterations)
        else:
            specs = mcp.tools_for_servers(agent_def.mcp_server_names)
            allowed = [s for s in specs if policy.can_call_tool(caller, s.namespaced_name).allow]
            tool_defs = [
                {"name": s.namespaced_name, "description": s.description, "input_schema": s.input_schema}
                for s in allowed
            ]
            bound = llm.bind_tools(tool_defs) if tool_defs else llm

        response = await bound.ainvoke([SystemMessage(content=system_text), *state["messages"]])
        return {"messages": [response]}

    async def execute_tools_node(state: AgentState) -> dict:
        caller = Caller.from_meta(state["caller"])
        last = state["messages"][-1]
        writer = get_stream_writer()
        outputs: list[ToolMessage] = []
        for call in last.tool_calls:
            name, args, call_id = call["name"], call["args"], call["id"]
            decision = policy.can_call_tool(caller, name)
            if not decision.allow:
                writer({"type": "access_denied", "agent": agent_def.agent_id, "tool": name, "reason": decision.reason})
                outputs.append(
                    ToolMessage(content=f"DENIED: {decision.reason}", tool_call_id=call_id, name=name)
                )
                continue

            approval = await needs_approval(agent_def.agent_id, name, args, store, caller)
            if approval.required:
                review_id = store.next_id("reviews", "REV")
                task_text = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
                await store.insert(
                    "reviews",
                    {
                        "id": review_id,
                        "status": "pending",
                        "agent": agent_def.agent_id,
                        "tool": name,
                        "args": args,
                        "reason": approval.reason,
                        "approver_roles": list(approval.approver_roles),
                        "requester": caller.user_id,
                        "caller_meta": caller.to_meta(),
                        "task": task_text,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "decided_at": None,
                        "approver": None,
                        "comment": None,
                        "result": None,
                    },
                )
                writer(
                    {
                        "type": "review_required",
                        "agent": agent_def.agent_id,
                        "tool": name,
                        "review_id": review_id,
                        "reason": approval.reason,
                    }
                )
                outputs.append(
                    ToolMessage(
                        content=(
                            f"PENDING_APPROVAL: this action has NOT been performed yet. "
                            f"Sent for approval as {review_id} ({approval.reason})."
                        ),
                        tool_call_id=call_id,
                        name=name,
                    )
                )
                continue

            writer({"type": "tool_call", "agent": agent_def.agent_id, "tool": name, "args": args})
            try:
                result = await mcp.call_tool(name, args, caller)
                text = _tool_result_text(result)
                if result.is_error:
                    text = f"ERROR: {text}"
            except Exception as e:  # noqa: BLE001 - surface any MCP failure to the LLM as a tool error
                text = f"ERROR: {e}"
            writer({"type": "tool_result", "agent": agent_def.agent_id, "tool": name, "result": text})
            outputs.append(ToolMessage(content=text, tool_call_id=call_id, name=name))
        return {"messages": outputs, "iterations": state.get("iterations", 0) + 1}

    def route_after_llm(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "execute_tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("trim_memory", make_trim_node(max_turns))
    graph.add_node("agent_llm", agent_llm_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_edge(START, "trim_memory")
    graph.add_edge("trim_memory", "agent_llm")
    graph.add_conditional_edges("agent_llm", route_after_llm, {"execute_tools": "execute_tools", END: END})
    graph.add_edge("execute_tools", "agent_llm")
    return graph.compile(checkpointer=checkpointer)
