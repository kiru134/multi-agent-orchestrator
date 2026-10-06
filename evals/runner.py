"""Drives the real orchestrator graph through each eval case and scores the trace.

Builds its own isolated Runtime (fresh store, fresh MCP connections, fresh agent auth) so an
eval run never touches whatever store/threads a live server or CLI session is using -
`POST /evals/run` can be hit against a running server without corrupting its demo data.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage, HumanMessage

from mas.config import settings_config
from mas.graphs.memory import ORCHESTRATOR, thread_config
from mas.paths import SEED_DIR, ROOT
from mas.runtime import Runtime, build_runtime
from evals.models import CaseResult, CaseTrace, CategorySummary, EvalCase, RunReport
from evals.scorers import score_case

DEFAULT_DATASET = ROOT / "evals" / "datasets" / "queries.yaml"


def load_cases(path: Path = DEFAULT_DATASET) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(c) for c in raw["cases"]]


async def _run_case(runtime: Runtime, case: EvalCase) -> CaseTrace:
    runtime.store.reset(SEED_DIR)
    caller = runtime.policy.get_caller(case.user)
    session_id = f"eval-{case.id}"
    cfg = thread_config(case.user, session_id, ORCHESTRATOR)
    cfg["recursion_limit"] = 50  # backstop only - see mas/graphs/orchestrator.py iterations cap

    trace = CaseTrace()
    start = time.monotonic()
    for turn in case.turns:
        try:
            async for mode, payload in runtime.orchestrator_graph.astream(
                {
                    "messages": [HumanMessage(content=turn)],
                    "caller": caller.to_meta(),
                    "user_id": case.user,
                    "session_id": session_id,
                    "iterations": 0,
                },
                config=cfg,
                stream_mode=["messages", "custom"],
            ):
                if mode != "custom":
                    continue
                kind = payload.get("type")
                if kind == "agent_start":
                    trace.agents_delegated.append(payload["agent"])
                elif kind == "tool_call":
                    trace.tools_called.append(payload["tool"])
                elif kind == "access_denied":
                    if "tool" in payload:
                        trace.tools_denied.append(payload["tool"])
                    else:
                        trace.agent_denials.append(payload["agent"])
                elif kind == "error":
                    trace.errors.append(f"{payload.get('agent')}: {payload.get('message')}")
        except Exception as e:  # noqa: BLE001 - record and keep scoring what we have
            trace.errors.append(f"orchestrator: {e!r}")
            break

    trace.latency_s = round(time.monotonic() - start, 3)
    state = await runtime.orchestrator_graph.aget_state(cfg)
    messages = state.values.get("messages", [])
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    trace.final_answer = last_ai.content if last_ai else ""
    trace.agents_delegated = list(dict.fromkeys(trace.agents_delegated))
    return trace


async def run_dataset(
    cases: list[EvalCase],
    backend: str | None = None,
    weights: dict[str, float] | None = None,
) -> RunReport:
    weights = weights or settings_config()["eval"]["weights"]
    runtime = await build_runtime(backend=backend)
    try:
        results: list[CaseResult] = []
        for case in cases:
            trace = await _run_case(runtime, case)
            results.append(score_case(case, trace, runtime.store, weights))
    finally:
        await runtime.aclose()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "Z"
    by_cat: dict[str, list[CaseResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    category_summaries = [
        CategorySummary(
            category=cat,
            count=len(rs),
            avg_score=round(sum(x.overall for x in rs) / len(rs), 1),
            pass_count=sum(1 for x in rs if x.verdict == "pass"),
            warn_count=sum(1 for x in rs if x.verdict == "warn"),
            fail_count=sum(1 for x in rs if x.verdict == "fail"),
        )
        for cat, rs in sorted(by_cat.items())
    ]

    return RunReport(
        run_id=run_id,
        started_at=run_id,
        backend=backend or "auto",
        weights=weights,
        results=results,
        avg_score=round(sum(r.overall for r in results) / len(results), 1) if results else 0.0,
        pass_count=sum(1 for r in results if r.verdict == "pass"),
        warn_count=sum(1 for r in results if r.verdict == "warn"),
        fail_count=sum(1 for r in results if r.verdict == "fail"),
        critical_violations=sum(1 for r in results if r.critical_violation),
        by_category=category_summaries,
    )
