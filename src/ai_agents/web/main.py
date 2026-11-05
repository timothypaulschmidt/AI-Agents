"""FastAPI backend: the agents' control page, their live log (SSE) and the operator's
reset/insert buttons."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..admin_client import AdminClient, AdminError
from ..agents import all_agents
from ..config import ROUTE_LABELS, Settings, load_settings
from ..runner import AgentRunner, RunConfig

log = logging.getLogger("ai_agents.web")
STATIC_DIR = Path(__file__).parent / "static"


class StartRequest(BaseModel):
    backend: Literal["ollama", "gemini"] = "ollama"
    model: str = Field(default="", max_length=100)
    tool_scope: Literal["all", "least"] = "all"
    dry_run: bool = False
    poll_seconds: float = Field(default=60, ge=1, le=3600)
    max_steps: int = Field(default=8, ge=1, le=30)


class ResetRequest(BaseModel):
    scenario: Literal["empty", "normal10", "poisoned10"] = "empty"
    full: bool = False


class InsertRequest(BaseModel):
    kind: Literal["normal", "poisoned"] = "normal"
    count: int = Field(default=1, ge=1, le=25)


class RouteRequest(BaseModel):
    route: Literal["direct", "faig_model", "faig_all", "faig"]   # "faig" = the old name for faig_all


def build_app(settings: Settings | None = None, client: httpx.AsyncClient | None = None,
              tools_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st = settings or load_settings()
        http = client or httpx.AsyncClient(timeout=httpx.Timeout(st.request_timeout, connect=5.0))
        app.state.settings = st
        app.state.client = http
        # FortiAIGate routes only: same settings, but may accept the gateway's self-signed lab certificate
        faig_http = client or httpx.AsyncClient(timeout=httpx.Timeout(st.request_timeout, connect=5.0),
                                                verify=st.faig_verify_tls)
        app.state.faig_client = faig_http
        app.state.route = "direct"
        app.state.admin = AdminClient(st, http)
        app.state.runners = {
            a.id: AgentRunner(a, st, http, get_route=lambda: app.state.route, tools_factory=tools_factory,
                              faig_client=faig_http)
            for a in all_agents().values()
        }
        yield
        for r in app.state.runners.values():
            await r.stop()
        if client is None:
            await http.aclose()
            await faig_http.aclose()

    app = FastAPI(title="AI Agents", lifespan=lifespan)

    def runner_for(request: Request, agent_id: str) -> AgentRunner:
        r = request.app.state.runners.get(agent_id)
        if r is None:
            raise HTTPException(404, f"unknown agent: {agent_id}")
        return r

    async def ollama_models(request: Request) -> list[str] | None:
        st: Settings = request.app.state.settings
        route = request.app.state.route
        ep = st.endpoints(route)
        try:
            if ep.gateway:
                # The gateway lists every provider's models under its own names. Show plain names,
                # and only the local ones (Gemini models are chosen separately).
                resp = await request.app.state.faig_client.get(
                    ep.ollama_base_url.rstrip("/") + "/v1/models",
                    headers={"Authorization": f"Bearer {st.faig_api_key}"}, timeout=httpx.Timeout(3.0))
                if resp.status_code == 200:
                    back = {wire: plain for plain, wire in st.faig_model_map.items()}
                    return [back[m["id"]] for m in resp.json().get("data", [])
                            if m.get("id") in back and back[m["id"]] not in st.gemini_models]
                return None
            resp = await request.app.state.client.get(
                ep.ollama_base_url.rstrip("/") + "/api/tags", timeout=httpx.Timeout(3.0))
            if resp.status_code == 200:
                return [m.get("name", "") for m in resp.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            pass
        return None

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/api/state")
    async def state(request: Request):
        st: Settings = request.app.state.settings
        return {
            "version": os.environ.get("APP_VERSION", "dev"),   # set at image build time; "dev" when run from source
            "route": {
                "current": request.app.state.route,
                "label": ROUTE_LABELS[request.app.state.route],
                "model_configured": st.faig_model_configured,
                "all_configured": st.faig_all_configured,
                "faig_configured": st.faig_configured,
            },
            "agents": [r.snapshot() for r in request.app.state.runners.values()],
            "ollama_models": await ollama_models(request),
            "default_ollama_model": st.ollama_model,
            "gemini_models": st.gemini_choices(request.app.state.route),
            "admin_enabled": bool(st.admin_token),
            "allow_parallel": st.allow_parallel,
        }

    @app.get("/api/stats")
    async def stats(request: Request):
        try:
            return await request.app.state.admin.stats()
        except AdminError as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    @app.post("/api/route")
    async def set_route(body: RouteRequest, request: Request):
        st: Settings = request.app.state.settings
        route = "faig_all" if body.route == "faig" else body.route
        problem = st.route_problem(route)
        if problem:
            raise HTTPException(400, problem)
        request.app.state.route = route
        for r in request.app.state.runners.values():
            r.emit("notice", message=f"Traffic: {ROUTE_LABELS[route]}. Takes effect on the next item.")
        return {"route": route, "label": ROUTE_LABELS[route]}

    @app.post("/api/agents/{agent_id}/start")
    async def start(agent_id: str, body: StartRequest, request: Request):
        runner = runner_for(request, agent_id)
        st: Settings = request.app.state.settings
        if runner.running:
            raise HTTPException(409, "already running")
        if not st.allow_parallel and any(r.running for r in request.app.state.runners.values()):
            raise HTTPException(409, "another agent is running (one at a time keeps the CPU free)")
        if body.backend == "gemini" and not st.gemini_api_key:
            raise HTTPException(400, "GEMINI_API_KEY is not set")
        model = body.model.strip() or (st.ollama_model if body.backend == "ollama"
                                       else (st.gemini_models[0] if st.gemini_models else ""))
        if body.backend == "ollama":
            names = await ollama_models(request)
            if names is not None and model not in names:
                raise HTTPException(400, f"{model} is not pulled in Ollama")
        runner.start(RunConfig(body.backend, model, body.tool_scope, body.dry_run,
                               body.poll_seconds, body.max_steps))
        return runner.snapshot()

    @app.post("/api/agents/{agent_id}/stop")
    async def stop(agent_id: str, request: Request):
        runner = runner_for(request, agent_id)
        await runner.stop()
        return runner.snapshot()

    @app.post("/api/agents/{agent_id}/poke")
    async def poke(agent_id: str, request: Request):
        runner_for(request, agent_id).poke()
        return {"ok": True}

    @app.get("/api/agents/{agent_id}/events")
    async def events(agent_id: str, request: Request, after: int = 0):
        runner = runner_for(request, agent_id)

        async def gen():
            q = runner.subscribe()
            try:
                last = after
                for e in list(runner.events):          # replay what the page missed
                    if e["seq"] > last:
                        last = e["seq"]
                        yield f"data: {json.dumps(e)}\n\n"
                while True:
                    try:
                        e = await asyncio.wait_for(q.get(), 15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        if await request.is_disconnected():
                            return
                        continue
                    if e["seq"] > last:
                        last = e["seq"]
                        yield f"data: {json.dumps(e)}\n\n"
            finally:
                runner.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def admin_call(coro):
        try:
            return await coro
        except AdminError as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    @app.post("/api/admin/reset")
    async def reset(body: ResetRequest, request: Request):
        result = await admin_call(request.app.state.admin.reset(body.scenario, body.full))
        if not isinstance(result, JSONResponse):
            for r in request.app.state.runners.values():
                r.emit("notice", message=f"Data reset: {body.scenario}{' (full rebuild)' if body.full else ''}.")
                r.poke()
        return result

    @app.post("/api/admin/insert")
    async def insert(body: InsertRequest, request: Request):
        result = await admin_call(request.app.state.admin.insert(body.kind, body.count))
        if not isinstance(result, JSONResponse):
            for r in request.app.state.runners.values():
                r.emit("notice", message=f"Inserted {body.count} {body.kind} ticket(s): ids {result.get('inserted_ids')}.")
                r.poke()
        return result

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = build_app()
