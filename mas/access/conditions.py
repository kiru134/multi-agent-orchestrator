"""Conditional-approval predicates, referenced by name from config/agents.yaml
(`approval: conditional, condition: <name>`). Each takes the tool call args, the store, and
the caller, and returns True when the call should be routed to human review.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from mas.access.models import Caller
from mas.store.document_store import DocumentStore

Condition = Callable[[dict[str, Any], DocumentStore, Caller], Awaitable[bool]]

SCALE_REPLICA_THRESHOLD = 5


async def scale_above_threshold(args: dict[str, Any], store: DocumentStore, caller: Caller) -> bool:
    try:
        return int(args.get("replicas", 0)) > SCALE_REPLICA_THRESHOLD
    except (TypeError, ValueError):
        # Malformed input - fail safe and require a human to look at it.
        return True


async def vm_over_quota(args: dict[str, Any], store: DocumentStore, caller: Caller) -> bool:
    team = args.get("team") or caller.team
    quota = await store.get("quotas", team)
    if quota is None:
        return True  # unknown team quota - don't let it slide
    count = int(args.get("count", 1))
    return quota["vm_used"] + count > quota["vm_limit"]


CONDITIONS: dict[str, Condition] = {
    "scale_above_threshold": scale_above_threshold,
    "vm_over_quota": vm_over_quota,
}
