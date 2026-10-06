"""The HTTP API: login, a streaming /chat endpoint (SSE), thread introspection, and agent
auth status. Everything goes through one shared Runtime built at startup."""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from mas.access.models import Caller
from mas.auth.users import LoginError, login, resolve_caller
from mas.config import settings_config
from mas.graphs.memory import ORCHESTRATOR, get_checkpointer, thread_config, thread_id
from mas.api.serialize import messages_to_list
from mas.runtime import Runtime, build_runtime

logger = logging.getLogger("mas.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = getattr(app.state, "backend_override", None)
    app.state.runtime = await build_runtime(backend=backend)
    yield
    await app.state.runtime.aclose()


app = FastAPI(title="Multi-Agent Orchestrator (prototype)", lifespan=lifespan)


def get_runtime() -> Runtime:
    return app.state.runtime


def get_caller(authorization: str | None = Header(default=None)) -> Caller:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "expected 'Authorization: Bearer <token>'")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return resolve_caller(token, get_runtime().policy)
    except LoginError as e:
        raise HTTPException(401, str(e)) from e


class LoginRequest(BaseModel):
    user_id: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user_id: str
    team: str
    roles: list[str]


@app.post("/auth/login", response_model=LoginResponse)
def auth_login(req: LoginRequest):
    try:
        token = login(req.user_id, req.password)
    except LoginError as e:
        raise HTTPException(401, str(e)) from e
    caller = get_runtime().policy.get_caller(req.user_id)
    return LoginResponse(token=token, user_id=caller.user_id, team=caller.team, roles=list(caller.roles))


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest, caller: Caller = Depends(get_caller)):
    runtime = get_runtime()
    runtime.session_registry.touch(caller.user_id, req.session_id, ORCHESTRATOR)
    cfg = thread_config(caller.user_id, req.session_id, ORCHESTRATOR)
    # The real cap on delegation rounds is OrchestratorState.iterations vs.
    # max_delegation_rounds (see mas/graphs/orchestrator.py) - this is just a generous
    # backstop against LangGraph's own hard recursion ceiling.
    cfg["recursion_limit"] = 50
    # InMemorySaver has no locking of its own - without this, two concurrent requests to the
    # same session (a double-submit, a retried request) could race and corrupt this
    # thread's checkpointed state. See ThreadLockRegistry.
    lock = await runtime.thread_locks.lock_for(thread_id(caller.user_id, req.session_id, ORCHESTRATOR))

    async def event_stream():
        async with lock:
            try:
                async for mode, payload in runtime.orchestrator_graph.astream(
                    {
                        "messages": [HumanMessage(content=req.message)],
                        "caller": caller.to_meta(),
                        "user_id": caller.user_id,
                        "session_id": req.session_id,
                        "iterations": 0,
                    },
                    config=cfg,
                    stream_mode=["messages", "custom"],
                ):
                    if mode == "messages":
                        chunk, meta = payload
                        if isinstance(chunk, AIMessage) and chunk.content and meta.get("langgraph_node") == "orchestrator_llm":
                            text = (
                                chunk.content
                                if isinstance(chunk.content, str)
                                else "".join(b.get("text", "") for b in chunk.content if isinstance(b, dict))
                            )
                            if text:
                                yield _sse({"type": "token", "agent": "orchestrator", "text": text})
                    elif mode == "custom":
                        kind = payload.get("type", "info")
                        source = payload.get("agent", "orchestrator")
                        rest = {k: v for k, v in payload.items() if k not in ("type", "agent")}
                        yield _sse({"type": kind, "agent": source, **rest})

                        for agent_touched in _agents_from_event(payload):
                            runtime.session_registry.touch(caller.user_id, req.session_id, agent_touched)

                state = await runtime.orchestrator_graph.aget_state(cfg)
                messages = state.values.get("messages", [])
                last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
                final_text = last_ai.content if last_ai else ""
                yield _sse({"type": "done", "agent": "orchestrator", "answer": final_text})
            except Exception as e:  # noqa: BLE001 - surface to the client instead of a bare 500 mid-stream
                logger.exception("error during chat stream")
                yield _sse({"type": "error", "agent": "orchestrator", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _agents_from_event(payload: dict) -> list[str]:
    agent = payload.get("agent")
    return [agent] if agent and agent != "orchestrator" else []


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/sessions/{session_id}/threads")
async def session_threads(session_id: str, caller: Caller = Depends(get_caller)):
    runtime = get_runtime()
    out = {}
    for agent in {ORCHESTRATOR, *runtime.session_registry.agents_used(caller.user_id, session_id)}:
        graph = runtime.orchestrator_graph if agent == ORCHESTRATOR else runtime.subagent_graphs.get(agent)
        if graph is None:
            continue
        cfg = thread_config(caller.user_id, session_id, agent)
        state = await graph.aget_state(cfg)
        out[agent] = messages_to_list(state.values.get("messages", []))
    return out


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, caller: Caller = Depends(get_caller)):
    runtime = get_runtime()
    for agent in {ORCHESTRATOR, *runtime.session_registry.agents_used(caller.user_id, session_id)}:
        await get_checkpointer().adelete_thread(thread_id(caller.user_id, session_id, agent))
    runtime.session_registry.forget(caller.user_id, session_id)
    return {"deleted": True}


@app.get("/agents")
def list_agents(caller: Caller = Depends(get_caller)):
    runtime = get_runtime()
    out: dict = {"_transport_mode": runtime.transport_mode}
    for agent_id, adef in runtime.agent_defs.items():
        out[agent_id] = {
            "authenticated": runtime.agent_registry.is_authenticated(agent_id),
            "description": adef.description,
            "mcp_servers": adef.mcp_server_names,
            "tool_count": len(runtime.mcp.tools_for_servers(adef.mcp_server_names)),
            "accessible_to_you": runtime.policy.can_use_agent(caller, agent_id).allow,
            "discoverable_via_a2a": runtime.transport.has_agent(agent_id) if runtime.transport_mode == "a2a" else None,
        }
    return out


@app.get("/agents/{agent_id}/card")
def agent_card(agent_id: str, caller: Caller = Depends(get_caller)):
    """The actual A2A Agent Card the orchestrator discovered for this agent (only populated
    when the server is running with transport_mode="a2a")."""
    runtime = get_runtime()
    if not runtime.discovered_agents or agent_id not in runtime.discovered_agents:
        raise HTTPException(404, f"no discovered A2A card for '{agent_id}' (transport_mode={runtime.transport_mode})")
    return runtime.discovered_agents[agent_id].card.model_dump(exclude_none=True)


class ReviewDecisionRequest(BaseModel):
    approve: bool
    comment: str = ""


@app.get("/reviews")
async def list_reviews(status: str | None = None, caller: Caller = Depends(get_caller)):
    """Reviews the caller either requested or is eligible to decide (never both - and
    never your own request, even if your role would otherwise qualify as an approver)."""
    runtime = get_runtime()

    def visible(doc: dict) -> bool:
        if status and doc["status"] != status:
            return False
        is_requester = doc["requester"] == caller.user_id
        can_decide = doc["requester"] != caller.user_id and caller.has_role(*doc["approver_roles"])
        return is_requester or can_decide

    docs = await runtime.store.find("reviews", visible)
    docs.sort(key=lambda d: d["created_at"], reverse=True)
    for d in docs:
        d["can_decide"] = (
            d["status"] == "pending" and d["requester"] != caller.user_id and caller.has_role(*d["approver_roles"])
        )
    return {"reviews": docs}


@app.post("/reviews/{review_id}/decision")
async def decide_review(review_id: str, req: ReviewDecisionRequest, caller: Caller = Depends(get_caller)):
    runtime = get_runtime()
    review = await runtime.store.get("reviews", review_id)
    if review is None:
        raise HTTPException(404, f"no such review '{review_id}'")
    if review["status"] != "pending":
        raise HTTPException(409, f"review '{review_id}' is already {review['status']}")
    if review["requester"] == caller.user_id:
        raise HTTPException(403, "you cannot approve or reject your own request")
    if not caller.has_role(*review["approver_roles"]):
        raise HTTPException(403, f"your role doesn't permit deciding this - needs one of {review['approver_roles']}")

    now = datetime.now(timezone.utc).isoformat()
    if not req.approve:
        return await runtime.store.update(
            "reviews", review_id, {"status": "rejected", "approver": caller.user_id, "comment": req.comment, "decided_at": now}
        )

    original_caller = Caller.from_meta(review["caller_meta"])
    try:
        result = await runtime.mcp.call_tool(review["tool"], review["args"], original_caller)
        result_payload = result.structured_content if result.structured_content is not None else {"ok": not result.is_error}
        status = "rejected" if result.is_error else "approved"
    except Exception as e:  # noqa: BLE001 - record the failure on the review rather than 500ing
        result_payload = {"error": str(e)}
        status = "approved_but_failed"

    return await runtime.store.update(
        "reviews",
        review_id,
        {"status": status, "approver": caller.user_id, "comment": req.comment, "decided_at": now, "result": result_payload},
    )


class EvalRunRequest(BaseModel):
    case: str | None = None
    category: str | None = None


@app.post("/evals/run")
async def trigger_eval_run(req: EvalRunRequest, caller: Caller = Depends(get_caller)):
    """Runs the eval dataset against a *fresh, isolated* runtime on the same backend this
    server was started with - it never touches this server's live store/threads. Makes real
    LLM calls if the server is on a live backend, so this can take a while and isn't free."""
    from evals.report import save_report
    from evals.runner import load_cases, run_dataset

    cases = load_cases()
    if req.case:
        cases = [c for c in cases if c.id == req.case]
    if req.category:
        cases = [c for c in cases if c.category == req.category]
    if not cases:
        raise HTTPException(400, "no matching cases")

    backend = getattr(app.state, "backend_override", None)
    report = await run_dataset(cases, backend=backend)
    save_report(report)
    logger.info("eval run %s triggered by %s: avg=%s pass=%s warn=%s fail=%s", report.run_id, caller.user_id, report.avg_score, report.pass_count, report.warn_count, report.fail_count)
    return report


@app.get("/evals/latest")
def eval_latest():
    from evals.report import latest_report

    report = latest_report()
    if report is None:
        raise HTTPException(404, "no eval runs yet - POST /evals/run or `python -m evals.run`")
    return report


@app.get("/evals/runs")
def eval_runs():
    from evals.report import list_runs

    return {"runs": list_runs()}


@app.get("/evals/runs/{run_id}")
def eval_run_detail(run_id: str):
    from evals.report import RESULTS_DIR, load_report

    path = RESULTS_DIR / run_id / "results.json"
    if not path.exists():
        raise HTTPException(404, f"no such run '{run_id}'")
    return load_report(path)


def _cli_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["anthropic", "ollama"], default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = _cli_args()
    app.state.backend_override = args.backend
    server_cfg = settings_config()["server"]
    uvicorn.run(app, host=args.host or server_cfg["host"], port=args.port or server_cfg["port"])
