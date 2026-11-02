"""The agent loop.

    wait / poll  ->  find the next work item (plain code, not the model)
    claim it     ->  the model works it in steps, calling tools (capped)
    wrap up      ->  make sure the item is not left half-done, log the outcome

The model never runs anything itself: it asks, and this code decides whether to run the
request. That is where an agent's safety controls live: which tools exist for it, a step
cap, dry-run, and a stop button. Every event goes to a log the web page streams.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable

import httpx

from .agents.base import Agent, WorkItem
from .config import Settings
from .mcp_client import McpError, McpTools, ToolResult, load_tools
from .providers import OpenAICompatProvider, ProviderError, ToolCall, Usage

log = logging.getLogger("ai_agents.runner")

LOG_LIMIT = 1000          # events kept for the page and for late viewers
PREVIEW_CHARS = 1500      # how much of a tool result the page shows
TOOL_SCOPES = ("all", "least")


@dataclass
class RunConfig:
    backend: str = "ollama"        # "ollama" or "gemini"
    model: str = ""
    tool_scope: str = "all"        # "all" server tools, or "least" = just what the job needs
    dry_run: bool = False          # read tools run, everything else is only logged
    poll_seconds: float = 60.0     # idle wait between checks (the "slow loop")
    max_steps: int = 8             # model rounds allowed per work item


def _preview(text: str) -> str:
    if len(text) <= PREVIEW_CHARS:
        return text
    return text[:PREVIEW_CHARS] + f"\n... [{len(text) - PREVIEW_CHARS} more characters not shown]"


def _assistant_message(text: str, calls: list[ToolCall]) -> dict:
    entries = []
    for c in calls:
        entry: dict[str, Any] = {"id": c.id, "type": "function",
                                 "function": {"name": c.name, "arguments": c.arguments}}
        if c.extra:
            entry["extra_content"] = c.extra  # must go back unchanged (Gemini)
        entries.append(entry)
    return {"role": "assistant", "content": text, "tool_calls": entries}


class AgentRunner:
    def __init__(self, agent: Agent, settings: Settings, client: httpx.AsyncClient,
                 get_route: Callable[[], str] = lambda: "direct",
                 tools_factory: Callable[[str], Any] | None = None):
        self.agent = agent
        self.settings = settings
        self.client = client
        self.get_route = get_route
        # Tests replace this to talk to an in-process MCP server.
        self._tools_factory = tools_factory or self._connect_http
        self.config = RunConfig(poll_seconds=settings.poll_seconds, max_steps=settings.max_steps)
        self.status = "stopped"      # stopped | idle | working
        self.current: WorkItem | None = None
        self.counters = {"items_done": 0, "items_failed": 0, "steps": 0, "tool_calls": 0, "blocked": 0}
        self.events: deque[dict] = deque(maxlen=LOG_LIMIT)
        self._seq = 0
        self._subs: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._poke = asyncio.Event()
        self._seen: set[str] = set()

    # ---- event log -------------------------------------------------------------

    def emit(self, type_: str, **data: Any) -> dict:
        self._seq += 1
        event = {"seq": self._seq, "ts": round(time.time(), 3), "type": type_, **data}
        self.events.append(event)
        log.info("%s %s", self.agent.id, json.dumps(event, default=str)[:600])
        for q in list(self._subs):
            q.put_nowait(event)
        return event

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def snapshot(self) -> dict:
        return {
            "id": self.agent.id, "name": self.agent.name, "description": self.agent.description,
            "status": self.status, "config": asdict(self.config),
            "current": self.current.title if self.current else None,
            "counters": dict(self.counters), "last_seq": self._seq,
        }

    def _set_status(self, status: str) -> None:
        if status != self.status:
            self.status = status
            self.emit("status", status=status)

    # ---- control ---------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, config: RunConfig) -> None:
        if self.running:
            raise RuntimeError("already running")
        if config.tool_scope not in TOOL_SCOPES:
            raise ValueError(f"tool_scope must be one of {TOOL_SCOPES}")
        self.config = config
        self._seen = set()
        self.counters = {k: 0 for k in self.counters}
        self.emit("notice", message=(
            f"Started: {config.backend}/{config.model}, tools={config.tool_scope}, "
            f"dry_run={config.dry_run}, poll={config.poll_seconds:g}s, max_steps={config.max_steps}, "
            f"route={self.get_route()}"))
        self._task = asyncio.create_task(self._main())

    async def stop(self) -> None:
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def poke(self) -> None:
        """Skip the rest of the idle wait and look for work now."""
        self._poke.set()

    # ---- the loop --------------------------------------------------------------

    async def _connect_http(self, route: str) -> McpTools:
        return await load_tools(self.settings.endpoints(route).mcp_url, timeout=10.0)

    def _provider(self, route: str) -> OpenAICompatProvider:
        ep = self.settings.endpoints(route)
        if self.config.backend == "gemini":
            return OpenAICompatProvider("gemini", self.client,
                                        ep.gemini_base_url.rstrip("/") + "/chat/completions",
                                        self.config.model, self.settings.gemini_api_key)
        return OpenAICompatProvider("ollama", self.client,
                                    ep.ollama_base_url.rstrip("/") + "/v1/chat/completions",
                                    self.config.model)

    async def _main(self) -> None:
        try:
            while True:
                try:
                    worked = await self._cycle()
                except (McpError, ProviderError, httpx.HTTPError) as exc:
                    self.emit("error", message=str(exc))
                    worked = False
                if worked:
                    continue  # more work may be waiting: check again straight away
                self._set_status("idle")
                self._poke.clear()
                try:
                    await asyncio.wait_for(self._poke.wait(), self.config.poll_seconds)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            self.emit("notice", message="Stopped.")
            raise
        finally:
            self.current = None
            self.status = "stopped"
            self.emit("status", status="stopped")

    async def _cycle(self) -> bool:
        """One look at the queue. Returns True if an item was worked."""
        route = self.get_route()
        tools = await self._tools_factory(route)
        item = await self.agent.next_item(tools, self._seen)
        if item is None:
            return False
        self.current = item
        self._set_status("working")
        self.emit("item_start", item=item.id, title=item.title)
        claimed = False
        try:
            if not self.config.dry_run:
                await self.agent.start_item(tools, item)
                claimed = True
            self._seen.add(item.id)
            ok = await self._work(tools, item, route)
            outcome = "model finished"
            if not self.config.dry_run:
                outcome = await self.agent.finish_item(tools, item)
            self.counters["items_done" if ok else "items_failed"] += 1
            self.emit("item_done", item=item.id, outcome=outcome, complete=ok)
        except asyncio.CancelledError:
            if claimed:
                await self._release(tools, item)
            self._seen.discard(item.id)
            raise
        except (ProviderError, httpx.HTTPError) as exc:
            # The model could not be reached: put the item back and try again next cycle.
            self._seen.discard(item.id)
            if claimed:
                await self._release(tools, item)
            self.counters["items_failed"] += 1
            self.emit("error", message=f"{exc}. Item {item.id} put back; will retry after the idle wait.")
            return False
        finally:
            self.current = None
        return True

    async def _release(self, tools: McpTools, item: WorkItem) -> None:
        try:
            await self.agent.release_item(tools, item)
            self.emit("notice", message=f"Item {item.id} put back in the queue.")
        except Exception as exc:  # best effort
            self.emit("error", message=f"Could not put item {item.id} back: {exc}")

    # ---- one work item ---------------------------------------------------------

    def _offered(self, tools: McpTools) -> list[dict]:
        specs = tools.openai_tools()
        if self.config.tool_scope == "least":
            allowed = self.agent.least_privilege_tools
            specs = [s for s in specs if s["function"]["name"] in allowed]
        return specs

    async def _execute(self, tools: McpTools, name: str, args: str) -> ToolResult:
        """Run one tool the model asked for, applying the safety controls."""
        if self.config.tool_scope == "least" and name not in self.agent.least_privilege_tools:
            self.counters["blocked"] += 1
            self.emit("blocked", name=name, reason="tool is not in this agent's allowed set")
            return ToolResult(f"Tool not available: {name}", True)
        if self.config.dry_run and name not in self.agent.read_tools:
            self.counters["blocked"] += 1
            self.emit("blocked", name=name, reason="dry run: not executed")
            return ToolResult(f"(dry run) {name} was not executed.", False)
        return await tools.call(name, args)

    async def _work(self, tools: McpTools, item: WorkItem, route: str) -> bool:
        """Let the model work one item. True if it finished within the step cap."""
        provider = self._provider(route)
        offered = self._offered(tools)
        messages: list[dict] = [
            {"role": "system", "content": self.agent.system_prompt},
            {"role": "user", "content": item.prompt},
        ]
        for step in range(1, self.config.max_steps + 1):
            self.counters["steps"] += 1
            self.emit("step", item=item.id, step=step, of=self.config.max_steps)
            text, calls = "", []
            async for part in provider.stream(messages, offered):
                if isinstance(part, Usage):
                    self.emit("usage", prompt_tokens=part.prompt_tokens,
                              completion_tokens=part.completion_tokens)
                elif isinstance(part, ToolCall):
                    calls.append(part)
                else:
                    text += part
            if text.strip():
                self.emit("model", text=text.strip())
            if not calls:
                return True
            messages.append(_assistant_message(text, calls))
            for call in calls:
                try:
                    shown: Any = json.loads(call.arguments)
                except ValueError:
                    shown = call.arguments
                self.counters["tool_calls"] += 1
                self.emit("tool_call", id=call.id, name=call.name, arguments=shown)
                result = await self._execute(tools, call.name, call.arguments)
                self.emit("tool_result", id=call.id, name=call.name,
                          content=_preview(result.text), is_error=result.is_error)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result.text})
        self.emit("notice", message=f"Step cap ({self.config.max_steps}) reached on item {item.id}.")
        return False
