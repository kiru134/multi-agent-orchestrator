#!/usr/bin/env python3
"""A small REPL client for the orchestrator API: logs in, then streams /chat responses,
color-coded per agent so you can see delegation happen live.

Demo users/passwords (see config/policies.yaml): alice/alice123 (engineer), bob/bob123
(it_admin), carol/carol123 (employee), dave/dave123 (manager).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

import httpx
from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table

AGENT_COLORS = {
    "orchestrator": "bright_white",
    "infra": "cyan",
    "it_support": "yellow",
    "resource_alloc": "magenta",
}


def color_for(agent: str) -> str:
    return AGENT_COLORS.get(agent, "green")


def render(console: Console, payload: dict) -> None:
    t = payload.get("type")
    agent = payload.get("agent", "orchestrator")
    c = color_for(agent)

    if t == "token":
        if agent != "orchestrator":
            return  # sub-agent token-by-token text is noisy; its tool calls + agent_end answer are enough
        console.print(escape(payload.get("text", "")), end="", style=c, highlight=False)
    elif t == "agent_start":
        console.print(f"\n[{c}]→ delegating to {agent}: {escape(payload.get('task', ''))}[/{c}]")
    elif t == "tool_call":
        console.print(f"[{c}]    • {payload['tool']}({escape(json.dumps(payload.get('args', {})))})[/{c}]")
    elif t == "tool_result":
        text = str(payload.get("result", ""))[:300]
        console.print(f"[dim {c}]      -> {escape(text)}[/dim {c}]")
    elif t == "access_denied":
        console.print(f"[bold red]    ✗ denied ({agent}): {escape(payload.get('reason', ''))}[/bold red]")
    elif t == "review_required":
        console.print(
            f"[bold yellow]    ⏳ sent for approval: {payload.get('review_id')} "
            f"({escape(payload.get('reason', ''))})[/bold yellow]"
        )
    elif t == "agent_end":
        console.print(f"[{c}]← {agent}: {escape(str(payload.get('answer', '')))}[/{c}]")
    elif t == "error":
        console.print(f"[bold red]error ({agent}): {escape(str(payload.get('message', '')))}[/bold red]")
    elif t == "done":
        console.print()


def print_reviews(console: Console, reviews: list[dict]) -> None:
    if not reviews:
        console.print("[dim]no reviews[/dim]")
        return
    table = Table()
    for col in ("id", "status", "agent", "tool", "requester", "reason", "you can decide"):
        table.add_column(col)
    for r in reviews:
        style = {"pending": "yellow", "approved": "green", "rejected": "red"}.get(r["status"], "white")
        table.add_row(
            r["id"],
            f"[{style}]{r['status']}[/{style}]",
            r["agent"],
            r["tool"],
            r["requester"],
            r["reason"][:60],
            "yes" if r.get("can_decide") else "",
        )
    console.print(table)


async def run(base_url: str, user_id: str, password: str) -> None:
    console = Console()
    session_id = str(uuid.uuid4())
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        resp = await client.post("/auth/login", json={"user_id": user_id, "password": password})
        if resp.status_code != 200:
            console.print(f"[bold red]login failed: {resp.text}[/bold red]")
            return
        body = resp.json()
        token = body["token"]
        headers = {"Authorization": f"Bearer {token}"}
        console.print(f"[green]logged in as {user_id} (roles: {', '.join(body['roles'])})[/green]")
        console.print("[dim]commands: /threads, /agents, /reviews, /approve <id> [comment], /reject <id> <comment>, /exit[/dim]")

        while True:
            try:
                message = Prompt.ask("\n[bold]>[/bold]")
            except (EOFError, KeyboardInterrupt):
                break
            message = message.strip()
            if not message:
                continue
            if message in ("/exit", "/quit"):
                break
            if message == "/threads":
                r = await client.get(f"/sessions/{session_id}/threads", headers=headers)
                console.print_json(data=r.json())
                continue
            if message == "/agents":
                r = await client.get("/agents", headers=headers)
                console.print_json(data=r.json())
                continue
            if message == "/reviews":
                r = await client.get("/reviews", headers=headers)
                print_reviews(console, r.json()["reviews"])
                continue
            if message.startswith("/approve ") or message.startswith("/reject "):
                approve = message.startswith("/approve ")
                rest = message.split(" ", 1)[1].strip()
                review_id, _, comment = rest.partition(" ")
                r = await client.post(
                    f"/reviews/{review_id}/decision",
                    json={"approve": approve, "comment": comment},
                    headers=headers,
                )
                if r.status_code != 200:
                    console.print(f"[bold red]{r.status_code}: {r.text}[/bold red]")
                else:
                    d = r.json()
                    console.print(f"[green]{review_id} -> {d['status']}[/green]  result={d.get('result')}")
                continue

            async with client.stream(
                "POST", "/chat", json={"session_id": session_id, "message": message}, headers=headers
            ) as r:
                if r.status_code != 200:
                    await r.aread()
                    console.print(f"[bold red]request failed: {r.status_code} {r.text}[/bold red]")
                    continue
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    render(console, json.loads(line[len("data: ") :]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="demo user id (alice, bob, carol, dave)")
    parser.add_argument("--password", default=None, help="defaults to prompting")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    password = args.password or Prompt.ask("password", password=True)
    asyncio.run(run(args.base_url, args.user, password))


if __name__ == "__main__":
    main()
