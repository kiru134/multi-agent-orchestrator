"""Persists a RunReport to evals/results/<run_id>/results.json (+ a latest.json pointer)
and renders a terminal summary."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from mas.paths import ROOT
from evals.models import RunReport

RESULTS_DIR = ROOT / "evals" / "results"


def save_report(report: RunReport) -> Path:
    run_dir = RESULTS_DIR / report.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "results.json"
    body = report.model_dump_json(indent=2)
    path.write_text(body, encoding="utf-8")
    (RESULTS_DIR / "latest.json").write_text(body, encoding="utf-8")
    return path


def load_report(path: Path) -> RunReport:
    return RunReport.model_validate_json(path.read_text(encoding="utf-8"))


def latest_report() -> RunReport | None:
    path = RESULTS_DIR / "latest.json"
    return load_report(path) if path.exists() else None


def list_runs() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted((p.name for p in RESULTS_DIR.iterdir() if p.is_dir()), reverse=True)


def render_terminal(report: RunReport) -> None:
    console = Console()
    table = Table(title=f"Eval run {report.run_id}  (backend={report.backend})")
    for col in ("Case", "Category", "Verdict", "Score", "Notes"):
        table.add_column(col)
    verdict_style = {"pass": "green", "warn": "yellow", "fail": "red"}
    for r in sorted(report.results, key=lambda r: r.overall):
        notes = "; ".join(f"{k}: {v.detail}" for k, v in r.dimensions.items() if v.score < 1.0)
        style = verdict_style[r.verdict]
        table.add_row(r.case_id, r.category, f"[{style}]{r.verdict}[/{style}]", str(r.overall), notes[:140])
    console.print(table)
    console.print(
        f"\n[bold]Overall:[/bold] avg={report.avg_score}  "
        f"pass={report.pass_count} warn={report.warn_count} fail={report.fail_count}  "
        f"critical_violations={report.critical_violations}"
    )
    if report.by_category:
        cat_table = Table(title="By category")
        for col in ("Category", "Count", "Avg", "Pass/Warn/Fail"):
            cat_table.add_column(col)
        for c in report.by_category:
            cat_table.add_row(c.category, str(c.count), str(c.avg_score), f"{c.pass_count}/{c.warn_count}/{c.fail_count}")
        console.print(cat_table)
