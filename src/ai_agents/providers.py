"""LLM providers (copied from the AI Assistant; see README on sharing code later).

Gemini (via its OpenAI-compatible endpoint) and Ollama both speak the same
``/chat/completions`` streaming protocol, so a single class covers both. Only
the URL, model name and API key differ. Unlike the Assistant there is no failover
router here: an agent uses the model you picked, and its errors are shown.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

import httpx

# Chat messages. Plain ones have string values; with tools, assistant messages carry
# a "tool_calls" list and "tool" messages carry a result, so values are not all strings.
Messages = list[dict[str, Any]]


@dataclass(frozen=True)
class Usage:
    """Token counts reported by the provider after an answer."""

    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ToolCall:
    """The model asking for a tool to be run. Nothing has run yet.

    ``arguments`` is the JSON text exactly as the model wrote it (it may be invalid:
    small models get this wrong, so the caller must parse it defensively).
    ``extra`` carries provider-specific data that must be sent back unchanged with
    the tool result (Gemini's "thought signature" travels this way).
    """

    id: str
    name: str
    arguments: str
    extra: dict | None = None


class ProviderError(Exception):
    """A provider failed. ``status`` is the HTTP status, or 0 if the
    connection failed or timed out."""

    def __init__(self, provider: str, status: int, message: str):
        if status == 0:
            text = f"{provider} unreachable: {message}"
        else:
            text = f"{provider} returned HTTP {status}: {message}"
        super().__init__(text)
        self.provider = provider
        self.status = status

    @property
    def retryable(self) -> bool:
        """True if trying a different provider makes sense. Rate limits,
        outages and connection problems are retryable; bad requests and bad
        API keys are not (they should be fixed, not hidden by a fallback)."""
        return self.status == 0 or self.status == 429 or self.status >= 500


class Provider(Protocol):
    name: str
    model: str

    def stream(
        self, messages: Messages, tools: list[dict] | None = None
    ) -> AsyncIterator[str | Usage | ToolCall]: ...


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        client: httpx.AsyncClient,
        chat_url: str,
        model: str,
        api_key: str = "",
        wire_model: str = "",
    ):
        self.name = name
        self.client = client
        self.chat_url = chat_url
        self.model = model
        # The name sent in the request when it differs from the name shown (a gateway renames models)
        self.wire_model = wire_model or model
        self.api_key = api_key
        # Ask for token counts. If a server rejects the option, it is switched
        # off once and the request retried, so chat never breaks over it.
        self.include_usage = True

    async def stream(
        self, messages: Messages, tools: list[dict] | None = None
    ) -> AsyncIterator[str | Usage | ToolCall]:
        try:
            async for item in self._stream(messages, tools):
                yield item
        except ProviderError as exc:
            if not (self.include_usage and exc.status == 400 and "stream_options" in str(exc).lower()):
                raise
            self.include_usage = False
            async for item in self._stream(messages, tools):
                yield item

    async def _stream(
        self, messages: Messages, tools: list[dict] | None = None
    ) -> AsyncIterator[str | Usage | ToolCall]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.wire_model, "messages": messages, "stream": True}
        if self.include_usage:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
        # Tool calls arrive as fragments; collect them by index, emit them whole at the end.
        pending: dict[int, dict] = {}
        try:
            async with self.client.stream(
                "POST", self.chat_url, json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise ProviderError(self.name, resp.status_code, body[:300].strip())
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        prompt = usage.get("prompt_tokens")
                        completion = usage.get("completion_tokens")
                        if isinstance(prompt, int) and isinstance(completion, int):
                            yield Usage(prompt, completion)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    for fragment in delta.get("tool_calls") or []:
                        _merge_tool_fragment(pending, fragment)
                    text = delta.get("content")
                    if text:
                        yield text
        except httpx.TransportError as exc:
            raise ProviderError(self.name, 0, f"{type(exc).__name__}: {exc}") from exc
        for index in sorted(pending):
            entry = pending[index]
            if not entry["name"]:
                continue
            yield ToolCall(
                entry["id"] or f"call_{index}",
                entry["name"],
                entry["args"].strip() or "{}",
                entry["extra"],
            )


def _merge_tool_fragment(pending: dict[int, dict], fragment: dict) -> None:
    """Add one streamed ``tool_calls`` fragment to the calls collected so far.

    OpenAI-style servers send an ``index`` and spread the arguments over many chunks.
    Some servers (Gemini, Ollama) send each call whole, and may omit the index; a
    fragment that has an ``id`` then starts a new call, one without continues the last.
    """
    index = fragment.get("index")
    if not isinstance(index, int):
        if fragment.get("id") or not pending:
            index = len(pending)
        else:
            index = max(pending)
    entry = pending.setdefault(index, {"id": "", "name": "", "args": "", "extra": None})
    if fragment.get("id"):
        entry["id"] = fragment["id"]
    function = fragment.get("function") or {}
    if function.get("name"):
        entry["name"] = function["name"]
    args = function.get("arguments")
    if isinstance(args, str):
        entry["args"] += args
    elif isinstance(args, dict):  # some servers send an object, not JSON text
        entry["args"] = json.dumps(args)
    if isinstance(fragment.get("extra_content"), dict):
        entry["extra"] = fragment["extra_content"]
