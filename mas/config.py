"""Loads and caches the YAML config files under config/.

Everything here is read-only, process-wide config - not per-request state. Call the
`*_config()` functions; they cache after the first read. Use `reload_all()` in tests that
need to point at different config files (it clears the cache).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from mas.paths import CONFIG_DIR, PROMPTS_DIR


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@functools.lru_cache(maxsize=1)
def settings_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "settings.yaml")


@functools.lru_cache(maxsize=1)
def policies_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "policies.yaml")


@functools.lru_cache(maxsize=1)
def agents_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "agents.yaml")


@functools.lru_cache(maxsize=1)
def models_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "models.yaml")


@functools.lru_cache(maxsize=32)
def prompt_text(relative_or_name: str) -> str:
    """Load a prompt file. Accepts a path relative to repo root (as agents.yaml stores it)
    or a bare filename under config/prompts/."""
    p = Path(relative_or_name)
    if not p.is_absolute():
        candidate = CONFIG_DIR.parent / relative_or_name
        p = candidate if candidate.exists() else PROMPTS_DIR / relative_or_name
    return p.read_text(encoding="utf-8")


def reload_all() -> None:
    """Clear all config caches. Tests use this after writing temp config files."""
    settings_config.cache_clear()
    policies_config.cache_clear()
    agents_config.cache_clear()
    models_config.cache_clear()
    prompt_text.cache_clear()
