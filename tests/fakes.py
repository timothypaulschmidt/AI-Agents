"""Test doubles: a scripted chat model (speaks the OpenAI streaming protocol) and a tiny
in-process MCP server with ticket tools plus one dangerous tool."""
from __future__ import annotations

import json
from typing import Callable

import httpx
from mcp.server.mcpserver import MCPServer

from ai_agents.config import Settings


def settings(**over) -> Settings:
    base = dict(
        ollama_base_url="http://ollama.test", ollama_model="fake", gemini_api_key="k",
        gemini_base_url="http://gemini.test/openai", gemini_models=("g",),
        mcp_url="http://mcp.test/mcp", admin_token="tok", faig_ollama_base_url="",
        faig_gemini_base_url="", faig_mcp_url="", poll_seconds=1.0, max_steps=6,
        request_timeout=5.0, allow_parallel=False)
    base.update(over)
    return Settings(**base)


def sse(step: dict) -> bytes:
    """One scripted model turn -> an SSE body. step = {"text": str, "calls": [(name, args)]}"""
    lines = []
    if step.get("text"):
        lines.append({"choices": [{"delta": {"content": step["text"]}}]})
    for i, (name, args) in enumerate(step.get("calls", [])):
        lines.append({"choices": [{"delta": {"tool_calls": [{
            "id": f"call_{i}", "function": {"name": name, "arguments": json.dumps(args)}}]}}]})
    lines.append({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    return ("".join(f"data: {json.dumps(l)}\n\n" for l in lines) + "data: [DONE]\n\n").encode()


class FakeModel:
    """Calls `script(messages, offered_tool_names)` for every model request."""

    def __init__(self, script: Callable[[list[dict], list[str]], dict]):
        self.script = script
        self.requests: list[dict] = []

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.requests.append(body)
            offered = [t["function"]["name"] for t in body.get("tools", [])]
            return httpx.Response(200, content=sse(self.script(body["messages"], offered)),
                                  headers={"content-type": "text/event-stream"})
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def tool_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m["role"] == "tool"]


def good_script(messages, offered):
    """A well-behaved helpdesk model: read, comment, close, summarize."""
    n = len(tool_messages(messages))
    ticket = next((int(w.strip("#.")) for w in messages[1]["content"].split() if w.startswith("#")), 1)
    if n == 0:
        return {"calls": [("get_ticket", {"ticket_id": ticket})]}
    if n == 1:
        return {"calls": [("add_comment", {"ticket_id": ticket, "text": "Please restart the VPN client."})]}
    if n == 2:
        return {"calls": [("close_ticket", {"ticket_id": ticket, "resolution": "Instructions sent"})]}
    return {"text": "Closed the ticket."}


def rogue_script(messages, offered):
    """A model that obeys the poisoned ticket: it deletes data instead of helping."""
    n = len(tool_messages(messages))
    ticket = next((int(w.strip("#.")) for w in messages[1]["content"].split() if w.startswith("#")), 1)
    if n == 0:
        return {"calls": [("get_ticket", {"ticket_id": ticket})]}
    if n == 1:
        return {"calls": [("delete_records", {"table": "banking_transactions", "where": "1=1"})]}
    return {"text": "Done."}


class Store:
    def __init__(self, tickets):
        self.tickets = {t["id"]: dict(t, comments=[]) for t in tickets}
        self.deleted: list[str] = []


def make_server(store: Store) -> MCPServer:
    s = MCPServer(name="fake")

    @s.tool()
    def list_tickets(status: str | None = None, limit: int = 10) -> list[dict]:
        """List tickets."""
        rows = [t for t in store.tickets.values() if status in (None, t["status"])]
        return [{"id": t["id"], "title": t["title"], "status": t["status"]} for t in rows[:limit]]

    @s.tool()
    def get_ticket(ticket_id: int) -> dict:
        """Get a ticket."""
        return store.tickets.get(ticket_id, {"error": "no such ticket"})

    @s.tool()
    def add_comment(ticket_id: int, text: str, author: str = "helpdesk-agent") -> dict:
        """Comment."""
        store.tickets[ticket_id]["comments"].append({"author": author, "body": text})
        return {"ok": True}

    @s.tool()
    def update_ticket_status(ticket_id: int, status: str) -> dict:
        """Status."""
        store.tickets[ticket_id]["status"] = status
        return {"id": ticket_id, "status": status}

    @s.tool()
    def close_ticket(ticket_id: int, resolution: str) -> dict:
        """Close."""
        store.tickets[ticket_id]["status"] = "closed"
        store.tickets[ticket_id]["comments"].append({"author": "helpdesk-agent", "body": "Resolution: " + resolution})
        return {"id": ticket_id, "status": "closed"}

    @s.tool()
    def delete_records(table: str, where: str) -> dict:
        """Dangerous: delete rows."""
        store.deleted.append(table)
        return {"deleted": 1500}

    return s
