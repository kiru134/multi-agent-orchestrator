"""Turns LangChain BaseMessage objects into plain JSON for the API/CLI."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def message_to_dict(m: BaseMessage) -> dict[str, Any]:
    if isinstance(m, HumanMessage):
        return {"role": "human", "content": m.content}
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": m.content}
    if isinstance(m, ToolMessage):
        return {"role": "tool", "name": m.name, "tool_call_id": m.tool_call_id, "content": m.content}
    if isinstance(m, AIMessage):
        out: dict[str, Any] = {"role": "ai", "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
        return out
    return {"role": m.type, "content": getattr(m, "content", "")}


def messages_to_list(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [message_to_dict(m) for m in messages]
