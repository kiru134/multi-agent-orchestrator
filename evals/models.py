"""Typed shapes for the eval dataset, a case's execution trace, and its scored result."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StoreAssertion(BaseModel):
    collection: str
    id: str
    field: str
    equals: Any = None


class Expectation(BaseModel):
    agents: list[str] = Field(default_factory=list)  # agents that must be delegated to
    tools_called: list[str] = Field(default_factory=list)  # namespaced tools that must run
    tools_forbidden: list[str] = Field(default_factory=list)  # namespaced tools that must NOT run
    denied: bool | None = None  # at least one access_denied event must (True) / must not (False) occur
    store: list[StoreAssertion] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)  # substrings (case-insensitive) in the final answer
    must_not_include: list[str] = Field(default_factory=list)
    max_steps: int | None = None  # soft cap on orchestrator_llm + agent_llm calls total


class EvalCase(BaseModel):
    id: str
    category: str
    user: str
    turns: list[str]
    expect: Expectation
    description: str = ""


class CaseTrace(BaseModel):
    agents_delegated: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    tools_denied: list[str] = Field(default_factory=list)  # tool-level denials, namespaced tool name
    agent_denials: list[str] = Field(default_factory=list)  # agent-level denials, agent id
    errors: list[str] = Field(default_factory=list)
    final_answer: str = ""
    llm_steps: int = 0
    latency_s: float = 0.0

    @property
    def denied(self) -> bool:
        return bool(self.tools_denied or self.agent_denials)


class DimensionScore(BaseModel):
    score: float  # 0..1, or -1 if not applicable (no expectation set for this dimension)
    detail: str


class CaseResult(BaseModel):
    case_id: str
    category: str
    user: str
    description: str
    overall: float  # 0..100
    verdict: str  # pass | warn | fail
    critical_violation: bool
    dimensions: dict[str, DimensionScore]
    trace: CaseTrace


class CategorySummary(BaseModel):
    category: str
    count: int
    avg_score: float
    pass_count: int
    warn_count: int
    fail_count: int


class RunReport(BaseModel):
    run_id: str
    started_at: str
    backend: str
    weights: dict[str, float]
    results: list[CaseResult]
    avg_score: float
    pass_count: int
    warn_count: int
    fail_count: int
    critical_violations: int
    by_category: list[CategorySummary]
