#!/usr/bin/env python3
"""CLI: python -m evals.run [--backend anthropic|ollama] [--case ID] [--category CAT]

Runs the sample queries in evals/datasets/queries.yaml against the live orchestrator graph
(a fresh, isolated runtime - this never touches a running server's store), scores each case,
prints a terminal summary, and saves evals/results/<run_id>/results.json (+ latest.json).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from evals.report import render_terminal, save_report
from evals.runner import load_cases, run_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["anthropic", "ollama"], default=None, help="default: auto-detect")
    parser.add_argument("--case", default=None, help="run only this case id")
    parser.add_argument("--category", default=None, help="run only this category")
    args = parser.parse_args()

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c.id == args.case]
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if not cases:
        print("no matching cases", file=sys.stderr)
        raise SystemExit(2)

    report = asyncio.run(run_dataset(cases, backend=args.backend))
    path = save_report(report)
    render_terminal(report)
    print(f"\nsaved to {path}")
    if report.critical_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
