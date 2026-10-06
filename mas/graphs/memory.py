"""Short-term memory: one shared checkpointer, thread-id scheme, a trim node factory, and
session bookkeeping for idle eviction.

Thread-id scheme: f"{user_id}:{session_id}:{agent}" where agent is "orchestrator" for the
main graph and the agent id (infra/it_support/resource_alloc) for a sub-agent graph. Because
each agent gets its own thread id, every agent's history is naturally isolated by the
checkpointer - a sub-agent's graph.astream never sees another agent's messages.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver

ORCHESTRATOR = "orchestrator"


def thread_id(user_id: str, session_id: str, agent: str) -> str:
    return f"{user_id}:{session_id}:{agent}"


def thread_config(user_id: str, session_id: str, agent: str) -> dict:
    return {"configurable": {"thread_id": thread_id(user_id, session_id, agent)}}


_checkpointer: InMemorySaver | None = None


def get_checkpointer() -> InMemorySaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = InMemorySaver()
    return _checkpointer


@dataclass
class ThreadLockRegistry:
    """Serializes concurrent requests to the same checkpointer thread.

    `InMemorySaver` (see get_checkpointer) has no locking of its own - it's a plain dict,
    and its own docstring says it's for debugging/testing only. Two concurrent requests to
    the same thread_id (a double-click, a retried request, two browser tabs on one session)
    can otherwise race: each full graph run does several checkpoint writes over its
    lifetime, last-write-wins on the *whole* checkpoint rather than merging, so an
    interleaved second run can silently clobber the first's state. One lock per thread_id,
    created lazily - same pattern as DocumentStore's per-collection locks.
    """

    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    _registry_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def lock_for(self, thread_id: str) -> asyncio.Lock:
        async with self._registry_lock:
            if thread_id not in self._locks:
                self._locks[thread_id] = asyncio.Lock()
            return self._locks[thread_id]


def reset_checkpointer() -> None:
    """Tests / eval runs between cases use this to start with a clean slate."""
    global _checkpointer
    _checkpointer = InMemorySaver()


def trim_history(messages: list[BaseMessage], max_turns: int) -> list[RemoveMessage]:
    """Keeps the most recent `max_turns` human turns (and everything after the earliest one
    kept). Cutting exactly on a HumanMessage boundary guarantees an AIMessage's tool_calls
    and their ToolMessages are never split apart, since tool exchanges only ever appear
    between one HumanMessage and the next."""
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= max_turns:
        return []
    cutoff = human_indices[-max_turns]
    return [RemoveMessage(id=m.id) for m in messages[:cutoff] if m.id is not None]


def make_trim_node(max_turns: int):
    """Returns a graph node function `(state) -> {"messages": [...]}` that trims
    `state["messages"]` to the configured window. Shared by the orchestrator graph and the
    sub-agent graph builder since both state schemas carry a `messages` key."""

    def _trim(state: dict) -> dict:
        removals = trim_history(state.get("messages", []), max_turns)
        return {"messages": removals} if removals else {}

    return _trim


@dataclass
class SessionRegistry:
    """Tracks (user_id, session_id) activity for idle-TTL cleanup. Doesn't store any
    conversation content itself - the checkpointer owns that; this just knows which threads
    belong to a session so they can be deleted together."""

    _last_seen: dict[tuple[str, str], float] = field(default_factory=dict)
    _agents_used: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    def touch(self, user_id: str, session_id: str, agent: str) -> None:
        key = (user_id, session_id)
        self._last_seen[key] = time.time()
        self._agents_used.setdefault(key, set()).add(agent)

    def agents_used(self, user_id: str, session_id: str) -> set[str]:
        return set(self._agents_used.get((user_id, session_id), set()))

    def active_sessions(self) -> list[tuple[str, str]]:
        return list(self._last_seen.keys())

    def idle_sessions(self, idle_ttl_seconds: float) -> list[tuple[str, str]]:
        now = time.time()
        return [k for k, last in self._last_seen.items() if now - last > idle_ttl_seconds]

    def forget(self, user_id: str, session_id: str) -> None:
        key = (user_id, session_id)
        self._last_seen.pop(key, None)
        self._agents_used.pop(key, None)

    async def evict_idle(self, checkpointer: InMemorySaver, idle_ttl_seconds: float) -> list[tuple[str, str]]:
        evicted = []
        for user_id, session_id in self.idle_sessions(idle_ttl_seconds):
            for agent in {ORCHESTRATOR, *self.agents_used(user_id, session_id)}:
                await checkpointer.adelete_thread(thread_id(user_id, session_id, agent))
            self.forget(user_id, session_id)
            evicted.append((user_id, session_id))
        return evicted
