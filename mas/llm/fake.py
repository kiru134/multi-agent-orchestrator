"""A scripted chat model for offline tests and the `--backend fake` eval run.

Two independent scripts:
  - `responses`: AIMessage objects cycled through on every `invoke`/`stream` call (used for
    the normal bind_tools() chat loop - set `.tool_calls` on a message to script a tool call).
  - `structured_responses`: plain objects cycled through on every `with_structured_output()`
    call (used for the orchestrator's router node).

Both cycle back to the start after the last item, like langchain_core's own FakeListChatModel,
so a short script can drive a longer-than-scripted conversation in a pinch (tests should
still size the script to the exact number of calls they expect).
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field


class FakeToolChatModel(FakeMessagesListChatModel):
    structured_responses: list[Any] = Field(default_factory=list)
    structured_i: int = 0

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        # The fake model ignores the actual tool schemas - the script already encodes the
        # exact tool_calls each scripted AIMessage should carry.
        return self

    def with_structured_output(
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        def _next(_input: Any) -> Any:
            if not self.structured_responses:
                raise RuntimeError("FakeToolChatModel: with_structured_output called with an empty script")
            resp = self.structured_responses[self.structured_i]
            if self.structured_i < len(self.structured_responses) - 1:
                self.structured_i += 1
            else:
                self.structured_i = 0
            return resp

        return RunnableLambda(_next)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-chat-model"
