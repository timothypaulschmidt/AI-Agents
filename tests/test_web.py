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
    assert s["route"] == {"current": "direct", "label": "Direct", "model_configured": False,
                          "all_configured": False, "faig_configured": False}
    assert s["gemini_models"] == ["g"] and s["admin_enabled"] is True


def test_faig_routes_are_refused_until_configured():
    c, _ = make_client()
    with c:
        assert c.post("/api/route", json={"route": "faig_model"}).status_code == 400
        assert c.post("/api/route", json={"route": "faig_all"}).status_code == 400
        assert c.post("/api/route", json={"route": "faig"}).status_code == 400   # old name
        assert c.post("/api/route", json={"route": "direct"}).status_code == 200


def test_model_only_route_needs_just_the_model_address():
    c, _ = make_client(settings(faig_ollama_base_url="http://faig/ollama", faig_api_key="k"))
    with c:
        assert c.post("/api/route", json={"route": "faig_model"}).json()["route"] == "faig_model"
        st = c.get("/api/state").json()["route"]
        assert st["current"] == "faig_model" and st["label"] == "Model via FortiAIGate"
        assert c.post("/api/route", json={"route": "faig_all"}).status_code == 400   # MCP address still empty
        assert "FAIG_MCP_SERVER_URL" in c.post("/api/route", json={"route": "faig_all"}).json()["detail"]


def test_model_and_mcp_route():
    c, _ = make_client(settings(faig_ollama_base_url="http://faig/ollama", faig_mcp_url="http://faig/mcp", faig_api_key="k"))
    with c:
        assert c.post("/api/route", json={"route": "faig_all"}).json()["route"] == "faig_all"
        assert c.post("/api/route", json={"route": "faig"}).json()["route"] == "faig_all"   # old name still works
        assert c.get("/api/state").json()["route"]["current"] == "faig_all"
        assert c.post("/api/route", json={"route": "direct"}).json()["route"] == "direct"


def test_endpoints_per_route():
    st = settings(faig_ollama_base_url="http://faig/ollama", faig_gemini_base_url="http://faig/gemini",
                  faig_mcp_url="http://faig/mcp")
    assert st.endpoints("direct").mcp_url == "http://mcp.test/mcp"
    assert st.endpoints("faig_model").ollama_base_url == "http://faig/ollama"
    assert st.endpoints("faig_model").mcp_url == "http://mcp.test/mcp"        # MCP stays direct
    assert st.endpoints("faig_all").mcp_url == "http://faig/mcp"


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


def test_state_reports_the_release_version(monkeypatch):
    monkeypatch.setenv("APP_VERSION", "v9.9.9")
    c, _ = make_client()
    with c:
        assert c.get("/api/state").json()["version"] == "v9.9.9"
    monkeypatch.delenv("APP_VERSION")
    c, _ = make_client()
    with c:
        assert c.get("/api/state").json()["version"] == "dev"


def test_gateway_route_needs_the_key():
    c, _ = make_client(settings(faig_ollama_base_url="http://faig/ollama"))
    with c:
        r = c.post("/api/route", json={"route": "faig_model"})
        assert r.status_code == 400 and "FAIG_API_KEY" in r.json()["detail"]


def test_wire_names_and_gemini_choices():
    from ai_agents.config import parse_model_map
    m = parse_model_map("qwen3:8b=ollama-qwen3-8b,gemini-3.6-flash=google-gemini-3.6-flash,bad")
    assert m == {"qwen3:8b": "ollama-qwen3-8b", "gemini-3.6-flash": "google-gemini-3.6-flash"}
    st = settings(faig_ollama_base_url="http://faig/ai/openai", faig_gemini_base_url="http://faig/ai/openai/v1",
                  faig_api_key="k", faig_model_map=m, gemini_api_key="g",
                  gemini_models=("gemini-3.6-flash", "gemini-3.5-flash"))
    assert st.wire_name("direct", "qwen3:8b") == "qwen3:8b"
    assert st.wire_name("faig_model", "qwen3:8b") == "ollama-qwen3-8b"
    assert st.gemini_choices("direct") == ["gemini-3.6-flash", "gemini-3.5-flash"]
    assert st.gemini_choices("faig_model") == ["gemini-3.6-flash"]      # only mapped ones
    assert st.endpoints("faig_model").gateway and not st.endpoints("direct").gateway
