"""The event shape streamed out over SSE (and consumed by the CLI / evals trace)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    type: str  # token | agent_start | agent_end | tool_call | tool_result | access_denied | error | done
    agent: str  # "orchestrator" or a sub-agent id
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "agent": self.agent, **self.data}
