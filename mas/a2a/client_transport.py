"""The orchestrator's A2A-based transport: discovers agents by fetching their real Agent
Cards (replacing config/agents.yaml's `description` as the source of truth for what the
orchestrator is told an agent can do) and calls them over the real A2A JSON-RPC protocol.

Runs over an ASGI loopback (`httpx.ASGITransport`) rather than a real TCP socket, so this
stays genuinely one process while still exercising the actual HTTP/A2A wire format - no
code path here is aware it isn't talking to a separately-deployed agent.

Scope note: unlike InProcessTransport, intermediate tool_call/tool_result/review_required
events are not forwarded live across the A2A boundary (A2A's streaming task/artifact-event
lifecycle is real extra protocol surface this prototype doesn't implement) - only
agent_start/agent_end/error. The sub-agent's own final answer still reports gated actions
and denials accurately (that's in the graph, unchanged); you just don't see it token by
token. A full per-agent conversation, including its tool calls, is still visible via its own
thread/checkpoint history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.client.helpers import create_text_message_object
from a2a.types import AgentCard, Message, Role
from fastapi import FastAPI

from mas.a2a.agent_card import agent_card_path
from mas.access.models import Caller
from mas.orchestrator.events import StreamEvent

LOOPBACK_BASE_URL = "http://a2a.local"


@dataclass(frozen=True)
class DiscoveredAgent:
    agent_id: str
    description: str
    skill_names: list[str]
    card: AgentCard


async def discover_agents(
    app: FastAPI, agent_ids: list[str], http_client: httpx.AsyncClient
) -> dict[str, DiscoveredAgent]:
    """Fetches each agent's real Agent Card (served by mas/a2a/server.py) over ASGI
    loopback - this is the actual discovery step, not a read of agents.yaml."""
    out: dict[str, DiscoveredAgent] = {}
    for agent_id in agent_ids:
        resolver = A2ACardResolver(http_client, LOOPBACK_BASE_URL, agent_card_path(agent_id))
        card = await resolver.get_agent_card()
        out[agent_id] = DiscoveredAgent(
            agent_id=agent_id,
            description=card.description,
            skill_names=[s.name for s in card.skills],
            card=card,
        )
    return out


def _message_text(message: Message) -> str:
    parts = []
    for p in message.parts:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


class A2AClientTransport:
    """Implements the same AgentTransport protocol as InProcessTransport, but calls agents
    over real A2A JSON-RPC instead of invoking their graphs directly in Python."""

    def __init__(self, http_client: httpx.AsyncClient, discovered: dict[str, DiscoveredAgent]) -> None:
        self._http_client = http_client
        self._discovered = discovered
        config = ClientConfig(httpx_client=http_client, streaming=False)
        factory = ClientFactory(config)
        self._clients = {agent_id: factory.create(d.card) for agent_id, d in discovered.items()}

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self._clients

    async def invoke_stream(
        self, agent_id: str, task: str, caller: Caller, user_id: str, session_id: str
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent("agent_start", agent_id, {"task": task})
        client = self._clients[agent_id]
        message = create_text_message_object(role=Role.user, content=task)
        answer = "(no answer produced)"
        try:
            async for event in client.send_message(
                message,
                request_metadata={"caller": caller.to_meta(), "user_id": user_id, "session_id": session_id},
            ):
                if isinstance(event, Message):
                    text = _message_text(event)
                    if text:
                        answer = text
                # (Task, Update) pairs aren't expected here - our executor always responds
                # with a single immediate Message, never opens a long-running Task.
        except Exception as e:  # noqa: BLE001 - surface the A2A call failure, don't crash the turn
            yield StreamEvent("error", agent_id, {"message": f"A2A call to '{agent_id}' failed: {e}"})
        yield StreamEvent("agent_end", agent_id, {"answer": answer})
