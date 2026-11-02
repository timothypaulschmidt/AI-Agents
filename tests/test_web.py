import json
import time

import httpx
from fastapi.testclient import TestClient

from ai_agents.mcp_client import load_tools
from ai_agents.web.main import build_app
from fakes import Store, good_script, make_server, settings, sse

TICKETS = [{"id": 1, "title": "VPN drops", "body": "x", "status": "open"}]


def make_client(st=None, store=None):
    store = store or Store(TICKETS)
    server = make_server(store)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "fake"}, {"name": "other"}]})
        if request.url.path.startswith("/admin/") or request.url.host == "mcp.test":
            if request.headers.get("authorization") != "Bearer tok":
                return httpx.Response(401, json={"error": "unauthorized"})
            if request.url.path == "/admin/stats":
                return httpx.Response(200, json={"tickets": {"open": 1, "in_progress": 0, "pending": 0, "closed": 0, "total": 1},
                                                 "poisoned_ticket_ids": [], "banking_transactions": 1500, "emails_in_outbox": 0})
            return httpx.Response(200, json={"inserted_ids": [9], "tickets": {}})
        body = json.loads(request.content)
        return httpx.Response(200, content=sse(good_script(body["messages"], [])),
                              headers={"content-type": "text/event-stream"})

    async def factory(route):
        return await load_tools(server)

    app = build_app(st or settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler)), factory)
    return TestClient(app), store


def wait_for(fn, seconds=8):
    end = time.time() + seconds
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.05)
    return False


def test_state_lists_agent_models_and_route():
    c, _ = make_client()
    with c:
        s = c.get("/api/state").json()
    assert s["agents"][0]["id"] == "helpdesk"
    assert s["ollama_models"] == ["fake", "other"]
    assert s["route"] == {"current": "direct", "faig_configured": False}
    assert s["gemini_models"] == ["g"] and s["admin_enabled"] is True


def test_faig_switch_is_refused_until_configured():
    c, _ = make_client()
    with c:
        assert c.post("/api/route", json={"route": "faig"}).status_code == 400
        assert c.post("/api/route", json={"route": "direct"}).status_code == 200
    c, _ = make_client(settings(faig_ollama_base_url="http://faig/ollama", faig_mcp_url="http://faig/mcp"))
    with c:
        assert c.post("/api/route", json={"route": "faig"}).json() == {"route": "faig"}
        assert c.get("/api/state").json()["route"]["current"] == "faig"


def test_start_work_and_stop():
    c, store = make_client()
    with c:
        r = c.post("/api/agents/helpdesk/start", json={"model": "fake", "poll_seconds": 30})
        assert r.status_code == 200
        assert wait_for(lambda: store.tickets[1]["status"] == "closed")
        assert wait_for(lambda: c.get("/api/state").json()["agents"][0]["status"] == "idle")
        assert c.post("/api/agents/helpdesk/start", json={"model": "fake"}).status_code == 409
        assert c.post("/api/agents/helpdesk/stop").json()["status"] == "stopped"


def test_start_validation():
    c, _ = make_client(settings(gemini_api_key=""))
    with c:
        assert c.post("/api/agents/helpdesk/start", json={"backend": "gemini"}).status_code == 400
        assert c.post("/api/agents/helpdesk/start", json={"model": "not-pulled"}).status_code == 400
        assert c.post("/api/agents/helpdesk/start", json={"tool_scope": "root"}).status_code == 422
        assert c.post("/api/agents/nope/start", json={}).status_code == 404


def test_event_stream_replays_the_log():
    c, store = make_client()
    with c:
        c.post("/api/agents/helpdesk/start", json={"model": "fake", "poll_seconds": 30})
        assert wait_for(lambda: store.tickets[1]["status"] == "closed")
        seen = []
        with c.stream("GET", "/api/agents/helpdesk/events?after=0") as resp:
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    seen.append(json.loads(line[5:])["type"])
                if "item_done" in seen:
                    break
        c.post("/api/agents/helpdesk/stop")
    assert "item_start" in seen and "tool_call" in seen and "item_done" in seen


def test_admin_proxy_and_missing_token():
    c, _ = make_client()
    with c:
        assert c.get("/api/stats").json()["tickets"]["open"] == 1
        assert c.post("/api/admin/insert", json={"kind": "normal", "count": 1}).json()["inserted_ids"] == [9]
        assert c.post("/api/admin/reset", json={"scenario": "bogus"}).status_code == 422
    c, _ = make_client(settings(admin_token=""))
    with c:
        r = c.get("/api/stats")
        assert r.status_code == 502 and "ADMIN_TOKEN" in r.json()["error"]


def test_index_page_is_served():
    c, _ = make_client()
    with c:
        r = c.get("/")
    assert r.status_code == 200 and "Action log" in r.text
