# Multi-Agent Orchestrator (MAS)

A prototype multi-agent "company assistant" built with **LangGraph**. A user talks to a
single **Orchestrator** agent, which routes and delegates work to specialist sub-agents —
**Infra**, **IT Support**, and **Resource Allocation** — each running as its own
LangGraph state machine, with RBAC enforced on every step and streaming output end to end.

## Why this exists

This started as a design exercise: what does a "real" multi-agent system need beyond a
router and a couple of tools? The answer ended up including access control per agent *and*
per tool, agent-to-agent authentication, bounded short-term memory, tiered model selection,
human-in-the-loop review for risky actions, and a way to actually measure whether the whole
thing is working. This repo is that system, scaled down to a single process and seeded with
fake company data so it can be run and evaluated end to end without any real infrastructure
behind it.

## Features

- **RBAC everywhere** — every request is checked against a role→agent→tool policy
  (`config/policies.yaml`) before it reaches a tool.
- **Agent-to-agent auth** — sub-agents complete a handshake with the orchestrator and hold a
  short-lived session JWT before they can be delegated any work.
- **Isolated memory per agent** — each agent gets its own LangGraph thread and short-term
  memory window; nothing leaks across agent boundaries.
- **Streaming throughout** — tokens and lifecycle events stream from sub-agents through the
  orchestrator to the client over SSE.
- **Pluggable LLMs** — a local model via **Ollama** and/or the **Anthropic** API, selected
  per task with tiered fallback (light / standard / heavy) when a tier's model isn't
  available.
- **Human-in-the-loop review** — tool calls tagged as high-risk in `agents.yaml` pause the
  graph (via LangGraph `interrupt()`) until a human approves or rejects them.
- **Tool access via MCP** — sub-agents call tools through mock MCP servers backed by a
  shared in-memory document store seeded from `data/seed/*.json`.
- **Evaluation harness** — a dataset of sample queries with expected outcomes, scored on
  routing, tool selection, access control, data leakage, and more.

## Architecture

```
CLI ──login──▶ /auth/login (user JWT)
CLI ──SSE───▶ /chat ─▶ Orchestrator graph  (thread_id = {user}:{session}:orchestrator)
                         │ delegate node: agent-level RBAC + agent session-token check
                         │ invokes sub-agent graph via AgentTransport, forwards its stream
                         ▼
        Infra graph / ITSupport graph / ResourceAlloc graph   (thread_id = {user}:{session}:{agent})
                         │ authorize_tools node: per-tool RBAC
                         │ execute_tools node: MCP call_tool (caller identity in _meta)
                         ▼
        MCPClientManager ─in-memory MCP transport─▶ infra-mcp / it-mcp / resources-mcp (FastMCP)
                         ▼
        DocumentStore (in-memory, seeded from data/seed/*.json)
```

Each sub-agent graph follows the same shape:

```
START → trim_memory → select_model → agent_llm ─(tool_calls)→ authorize_tools → execute_tools ─┬→ agent_llm
                                       ▲                                                        └(tool error)→ escalate ─┘
                                       └(final answer / max_iterations)→ finalize → END
```

## Project layout

```
mas/
  agents/        agent definitions (id, system prompt, mcp servers, max_iterations)
  graphs/         orchestrator + sub-agent LangGraph builders, memory/checkpointing
  llm/            model factory, tiered selector, fake model for tests/offline demo
  mcp/            MCP client manager + mock servers (infra, IT, resources)
  access/         RBAC policy engine + human-review approval flow
  auth/           user login (JWT) and agent-to-agent auth handshake
  orchestrator/   agent transport + streaming event types
  a2a/            Agent-to-Agent protocol server/client/executor
  api/            FastAPI app (SSE chat endpoint, eval report endpoint)
  store/          in-memory document store
config/
  settings.yaml   server, JWT, memory, and eval settings
  policies.yaml   roles → agents/tools RBAC matrix (demo users + PBKDF2 password hashes)
  agents.yaml     agent registry: id, mcp servers, tool risk tags
  models.yaml     provider + model tier catalogue
  prompts/        per-agent system prompts
data/seed/        fake company data (employees, services, incidents, tickets, ...)
evals/            evaluation harness: datasets, runner, scorers, terminal/JSON reports
tests/            pytest suite
cli.py            REPL client: logs in, streams /chat, color-codes output per agent
```

## Getting started

### Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

### Configure an LLM backend

Pick at least one:

- **Ollama** (local): install Ollama and pull a model, e.g. `ollama pull llama3.1`.
- **Anthropic**: `export ANTHROPIC_API_KEY=...`

The runtime auto-detects whichever backend is available; see `config/models.yaml` for the
tier → model mapping.

For anything beyond local experimentation, also set a real JWT signing secret:

```bash
export MAS_JWT_SECRET=<32+ random bytes>
```

(Without it, the dev default in `config/settings.yaml` is used — fine for local use only.)

### Run the API

```bash
python -m mas.api.app
```

### Run the CLI demo

```bash
python cli.py
```

Demo users (see `config/policies.yaml`): `alice`/`alice123` (engineer), `bob`/`bob123`
(it_admin), `carol`/`carol123` (employee), `dave`/`dave123` (manager).

### Run the tests

```bash
pytest
```

### Run the evaluation harness

```bash
python -m evals.run
```

Scores sample queries from `evals/datasets/queries.yaml` against a fresh runtime and saves
a report to `evals/results/<run_id>/`. Use `--backend`, `--case`, or `--category` to narrow
a run.

## Status

This is a prototype built to explore the architecture, not a production system — the mock
MCP servers, seed data, and demo credentials make that explicit. See `plan.md` for the
original design notes and the decisions behind each piece.
