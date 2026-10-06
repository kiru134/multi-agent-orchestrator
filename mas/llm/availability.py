"""Cheap, best-effort probe for which LLM providers are actually usable right now -
used once at startup so the ModelSelector doesn't try to route to a provider with no
credentials / no server running."""

from __future__ import annotations

import os

import httpx

from mas.config import models_config


async def detect_available_providers() -> set[str]:
    available: set[str] = set()
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        available.add("anthropic")
    base_url = models_config().get("ollama", {}).get("base_url", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                available.add("ollama")
    except httpx.HTTPError:
        pass
    return available
