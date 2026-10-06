"""Builds the orchestrator's compiled LangGraph StateGraph - a minimal state machine:

    START -> trim_memory -> orchestrator_llm --(tool_calls)--> delegate --> orchestrator_llm
                                              \\--(no tool_calls)--> END

`delegate` is where every requirement outside the LLM call itself lives: agent-level access
control, the agent-auth check (re-authenticating if the agent's session expired), and
invoking the sub-agent graph through the transport while forwarding its stream.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from mas.access.models import Caller
from mas.access.policy import PolicyEngine
from mas.agents.base import BaseAgent
from mas.auth.agent_auth import AgentAuthError, AgentSessionRegistry, ensure_authenticated
from mas.config import prompt_text
from mas.graphs.memory import make_trim_node
from mas.orchestrator.transport import AgentTransport

DELEGATE_PREFIX = "delegate_to_"


class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    caller: dict[str, Any]
    user_id: str
    session_id: str
    # Delegation rounds used *this turn* - reset to 0 by every caller on every new user
    # message (see mas/api/app.py, evals/runner.py), not carried over between turns.
    iterations: int


MAX_DELEGATION_REMINDER = (
    "You have used all {max_rounds} delegation rounds allowed for this turn. Do not call any "
    "more delegate_to_* tools. Reply to the user now with your best answer from what you've "
    "already learned - say plainly if part of the request couldn't be completed."
)


def build_orchestrator_graph(
    agent_defs: dict[str, BaseAgent],
    agent_registry: AgentSessionRegistry,
    policy: PolicyEngine,
    transport: AgentTransport,
    llm: BaseChatModel,
    checkpointer: InMemorySaver,
    max_turns: int,
    max_delegation_rounds: int = 6,
    system_prompt: str | None = None,
):
    system_prompt_template = system_prompt or prompt_text("config/prompts/orchestrator.md")

    def _available_agents(caller: Caller) -> list[BaseAgent]:
        out = []
        for agent_id, adef in agent_defs.items():
            if not agent_registry.is_authenticated(agent_id):
                continue
            if not policy.can_use_agent(caller, agent_id).allow:
                continue
            out.append(adef)
        return out

    async def orchestrator_llm_node(state: OrchestratorState) -> dict:
        caller = Caller.from_meta(state["caller"])
        available = _available_agents(caller)
        agent_list = "\n".join(f"- {a.agent_id}: {a.description}" for a in available) or "(none available)"
        system_text = system_prompt_template.format(agent_list=agent_list)

        if state.get("iterations", 0) >= max_delegation_rounds:
            bound = llm  # no delegate tools bound - force a direct answer
            system_text += "\n\n" + MAX_DELEGATION_REMINDER.format(max_rounds=max_delegation_rounds)
        else:
            tool_defs = [
                {
                    "name": f"{DELEGATE_PREFIX}{a.agent_id}",
                    "description": a.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "Self-contained task for the agent."}
                        },
                        "required": ["task"],
                    },
                }
                for a in available
            ]
            bound = llm.bind_tools(tool_defs) if tool_defs else llm

        response = await bound.ainvoke([SystemMessage(content=system_text), *state["messages"]])
        return {"messages": [response]}

    async def delegate_node(state: OrchestratorState) -> dict:
        caller = Caller.from_meta(state["caller"])
        last = state["messages"][-1]
        writer = get_stream_writer()

        async def invoke_one(agent_id: str, task: str) -> str:
            answer = "(no answer produced)"
            async for ev in transport.invoke_stream(agent_id, task, caller, state["user_id"], state["session_id"]):
                writer(ev.to_dict())
                if ev.type == "agent_end":
                    answer = ev.data.get("answer", answer)
            return answer

        # Pre-flight checks (agent exists, RBAC, auth) run eagerly and in order, rejecting in
        # place. Everything that passes is grouped by target agent: different agents run
        # concurrently - independent delegations in one turn (e.g. "laptop broken" + "need a
        # VM") no longer wait on each other - but two calls to the *same* agent are chained
        # sequentially, since they'd otherwise be two concurrent writers to that one agent's
        # checkpointer thread.
        results: list[str | None] = [None] * len(last.tool_calls)
        by_agent: dict[str, list[int]] = {}

        for i, call in enumerate(last.tool_calls):
            name = call["name"]
            agent_id = name.removeprefix(DELEGATE_PREFIX)

            if agent_id not in agent_defs or not transport.has_agent(agent_id):
                results[i] = f"ERROR: unknown agent '{agent_id}'"
                continue

            decision = policy.can_use_agent(caller, agent_id)
            if not decision.allow:
                writer({"type": "access_denied", "agent": agent_id, "reason": decision.reason})
                results[i] = f"DENIED: {decision.reason}"
                continue

            try:
                ensure_authenticated(agent_registry, agent_id)
            except AgentAuthError as e:
                writer({"type": "error", "agent": agent_id, "message": str(e)})
                results[i] = f"ERROR: agent '{agent_id}' is not authenticated: {e}"
                continue

            by_agent.setdefault(agent_id, []).append(i)

        async def run_agent_chain(agent_id: str, indices: list[int]) -> None:
            for i in indices:
                task_text = last.tool_calls[i]["args"].get("task", "")
                results[i] = await invoke_one(agent_id, task_text)

        await asyncio.gather(*(run_agent_chain(agent_id, idxs) for agent_id, idxs in by_agent.items()))

        outputs = [
            ToolMessage(content=results[i], tool_call_id=call["id"], name=call["name"])
            for i, call in enumerate(last.tool_calls)
        ]
        return {"messages": outputs, "iterations": state.get("iterations", 0) + 1}

    def route_after_llm(state: OrchestratorState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "delegate"
        return END

    graph = StateGraph(OrchestratorState)
    graph.add_node("trim_memory", make_trim_node(max_turns))
    graph.add_node("orchestrator_llm", orchestrator_llm_node)
    graph.add_node("delegate", delegate_node)
    graph.add_edge(START, "trim_memory")
    graph.add_edge("trim_memory", "orchestrator_llm")
    graph.add_conditional_edges("orchestrator_llm", route_after_llm, {"delegate": "delegate", END: END})
    graph.add_edge("delegate", "orchestrator_llm")
    return graph.compile(checkpointer=checkpointer)
