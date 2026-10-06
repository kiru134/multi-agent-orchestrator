"""Deterministic model-tier selection: the code decides which model handles a task, not an
LLM. Reads config/models.yaml by default; tests/CLI/evals can pass an explicit catalog and
force a single provider (e.g. "fake", or a user-forced --backend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mas.config import models_config

TIER_ORDER = ["light", "standard", "heavy"]


class NoProviderAvailable(Exception):
    pass


@dataclass(frozen=True)
class ModelChoice:
    tier: str
    provider: str
    model: str
    temperature: float
    reason: str


@dataclass
class ModelSelector:
    catalog: dict[str, Any] | None = None
    available_providers: set[str] | None = None  # None = assume every configured provider works
    force_provider: str | None = None  # e.g. "fake" in tests, or a user-forced --backend

    def __post_init__(self) -> None:
        if self.catalog is None:
            self.catalog = models_config()

    def _provider_order(self) -> list[str]:
        if self.force_provider:
            return [self.force_provider]
        order = list(self.catalog.get("provider_order", []))
        if self.available_providers is None:
            return order
        return [p for p in order if p in self.available_providers]

    def _provider_model(self, tier: str) -> tuple[str, str, float]:
        tier_cfg = self.catalog["tiers"][tier]
        for provider in self._provider_order():
            if provider in tier_cfg:
                m = tier_cfg[provider]
                return provider, m["model"], float(m.get("temperature", 0.0))
        raise NoProviderAvailable(
            f"no available provider for tier '{tier}' (tried {self._provider_order()}, "
            f"tier offers {list(tier_cfg.keys())})"
        )

    def select(self, task: str, *, complex_request: bool = False, high_risk: bool = False) -> ModelChoice:
        tier = self.catalog["tasks"][task]
        reason = f"task default ({task})"
        esc = self.catalog.get("escalation", {})
        if high_risk and esc.get("high_risk_tool"):
            tier = esc["high_risk_tool"]
            reason = "high_risk_tool"
        elif complex_request and esc.get("complex_request"):
            tier = esc["complex_request"]
            reason = "complex_request"
        provider, model, temperature = self._provider_model(tier)
        return ModelChoice(tier=tier, provider=provider, model=model, temperature=temperature, reason=reason)

    def escalate(self, choice: ModelChoice) -> ModelChoice:
        """One tier up, capped at heavy. Used after repeated tool errors/parse failures."""
        idx = TIER_ORDER.index(choice.tier)
        if idx >= len(TIER_ORDER) - 1:
            return choice
        new_tier = TIER_ORDER[idx + 1]
        provider, model, temperature = self._provider_model(new_tier)
        return ModelChoice(tier=new_tier, provider=provider, model=model, temperature=temperature, reason="on_failure")

    def for_tier(self, tier: str, reason: str = "explicit") -> ModelChoice:
        provider, model, temperature = self._provider_model(tier)
        return ModelChoice(tier=tier, provider=provider, model=model, temperature=temperature, reason=reason)
