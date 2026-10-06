"""Builds a LangChain chat model from a provider/model/temperature triple. Cached so the
same (provider, model, temperature) reuses one client. The "fake" provider is only ever
reached in tests - production/CLI paths always pick a real provider via ModelSelector."""

from __future__ import annotations

import functools

from langchain_core.language_models import BaseChatModel

from mas.config import models_config
from mas.llm.selector import ModelChoice


@functools.lru_cache(maxsize=32)
def _build(provider: str, model: str, temperature: float) -> BaseChatModel:
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Current Claude models (Opus 5.5, Sonnet 5.5, ...) run adaptive thinking by
        # default and reject a non-default temperature (400). Don't set it.
        return ChatAnthropic(model=model, max_tokens=4096)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = models_config().get("ollama", {}).get("base_url", "http://localhost:11434")
        return ChatOllama(model=model, temperature=temperature, base_url=base_url)
    if provider == "fake":
        from mas.llm.fake import FakeToolChatModel

        return FakeToolChatModel()
    raise ValueError(f"unknown LLM provider '{provider}'")


def get_chat_model(provider: str, model: str, temperature: float = 0.0) -> BaseChatModel:
    return _build(provider, model, temperature)


def get_chat_model_for(choice: ModelChoice) -> BaseChatModel:
    return _build(choice.provider, choice.model, choice.temperature)


def clear_cache() -> None:
    _build.cache_clear()
