"""Pulls the caller identity out of an MCP request's `_meta`, injected by the client on
every call_tool (see mas/mcp/client_manager.py). This is how mock servers do row-level
scoping (own tickets, own-team quota) - it never comes from the LLM, only from the agent
node that made the call.
"""

from __future__ import annotations

from mcp.server.mcpserver import Context

from mas.access.models import Caller


class MissingCaller(Exception):
    pass


def caller_from_ctx(ctx: Context) -> Caller:
    meta = ctx.request_context.meta
    caller_dict = dict(meta).get("caller") if meta else None
    if not caller_dict:
        raise MissingCaller("request carried no caller identity in _meta")
    return Caller.from_meta(caller_dict)
