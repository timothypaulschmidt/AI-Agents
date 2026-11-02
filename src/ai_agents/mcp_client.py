"""Talks to an MCP server (copied from the AI Assistant, plus call_data below).

`target` is normally the server's URL (Streamable HTTP, e.g. http://localhost:8000/mcp).
Tests pass an in-process MCPServer instead, which the same Client accepts.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from mcp import Client


class McpError(Exception):
    """The MCP server could not be reached or did not answer properly."""


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


async def list_tools(target: Any, timeout: float = 5.0) -> list[ToolInfo]:
    """Connect, ask the server what tools it has, disconnect."""

    async def _go() -> list[ToolInfo]:
        async with Client(target) as client:
            result = await client.list_tools()
            return [
                ToolInfo(t.name, t.description or "", dict(t.input_schema or {}))
                for t in result.tools
            ]

    try:
        return await asyncio.wait_for(_go(), timeout)
    except asyncio.TimeoutError as exc:
        raise McpError(f"MCP server did not answer within {timeout:g}s") from exc
    except Exception as exc:  # connection refused, bad URL, protocol error...
        raise McpError(f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Calling tools
# ---------------------------------------------------------------------------

import json

# What the model gets back from one tool call is capped, so one huge result can't
# fill the model's context window.
MAX_RESULT_CHARS = 20_000


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool = False


_DROP_KEYS = {"title", "additionalProperties", "$schema", "default"}


def clean_schema(node: Any, in_properties: bool = False) -> Any:
    """Simplify a JSON Schema for model providers that only accept a subset of it.

    Drops keys such as ``title`` and ``additionalProperties`` and turns an optional
    parameter (``anyOf: [X, null]``) into plain X. Names of parameters are never
    touched, even a parameter called ``title``.
    """
    if isinstance(node, list):
        return [clean_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if in_properties:
            out[key] = clean_schema(value)  # key is a parameter name: keep it
        elif key in _DROP_KEYS:
            continue
        else:
            out[key] = clean_schema(value, in_properties=(key == "properties"))
    if not in_properties and isinstance(out.get("anyOf"), list):
        variants = [v for v in out["anyOf"] if not (isinstance(v, dict) and v.get("type") == "null")]
        if len(variants) == 1 and isinstance(variants[0], dict):
            del out["anyOf"]
            out = {**variants[0], **out}
    return out


class McpTools:
    """The tools an MCP server offers, plus a way to call them.

    Each call opens a short-lived connection. That costs a handshake (cheap on a
    local network) and means no connection stays open while a model is thinking.
    """

    def __init__(self, target: Any, tools: list[ToolInfo], timeout: float = 30.0):
        self.target = target
        self.tools = tools
        self.timeout = timeout
        self.names = {t.name for t in tools}

    def openai_tools(self) -> list[dict]:
        """The tool list in the format chat models expect."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": clean_schema(t.input_schema) or {"type": "object", "properties": {}},
                },
            }
            for t in self.tools
        ]

    async def call(self, name: str, arguments: str | dict | None) -> ToolResult:
        """Run one tool. Never raises: a failure comes back as an error result, so the
        model sees what went wrong (and may try again) instead of the chat dying."""
        if name not in self.names:
            return ToolResult(f"Unknown tool: {name}", True)
        try:
            if isinstance(arguments, str):
                args = json.loads(arguments) if arguments.strip() else {}
            else:
                args = arguments or {}
        except json.JSONDecodeError as exc:
            return ToolResult(f"Invalid JSON arguments: {exc}", True)
        if not isinstance(args, dict):
            return ToolResult("Tool arguments must be a JSON object", True)

        async def _go():
            async with Client(self.target) as client:
                return await client.call_tool(name, args)

        try:
            res = await asyncio.wait_for(_go(), self.timeout)
        except asyncio.TimeoutError:
            return ToolResult(f"Tool call timed out after {self.timeout:g}s", True)
        except Exception as exc:
            return ToolResult(f"Tool call failed: {type(exc).__name__}: {exc}", True)

        parts = [getattr(b, "text", None) for b in res.content]
        text = "\n".join(p for p in parts if p) or "(no output)"
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n... [truncated, {len(text) - MAX_RESULT_CHARS} more characters]"
        return ToolResult(text, bool(res.is_error))

    async def call_data(self, name: str, args: dict) -> Any:
        """Call a tool and return parsed data. For the agent's own bookkeeping (picking
        the next ticket, setting a status), not for the model. Raises McpError on failure."""
        try:
            async with Client(self.target) as client:
                res = await asyncio.wait_for(client.call_tool(name, args), self.timeout)
        except Exception as exc:
            raise McpError(f"{name}: {type(exc).__name__}: {exc}") from exc
        if res.is_error:
            raise McpError(f"{name}: " + "; ".join(getattr(b, "text", "") or "" for b in res.content))
        return parse_result(res)


async def load_tools(target: Any, timeout: float = 5.0) -> McpTools:
    """Connect once, read the tool list, and return an object that can call them."""
    return McpTools(target, await list_tools(target, timeout))


def parse_result(res: Any) -> Any:
    """A tool result as Python data. Prefers the server's structured form; otherwise
    each text block is JSON (a list return arrives as one block per item)."""
    sc = getattr(res, "structured_content", None)
    if sc is not None:
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    blocks = []
    for b in res.content:
        text = getattr(b, "text", None)
        if text is None:
            continue
        try:
            blocks.append(json.loads(text))
        except ValueError:
            blocks.append(text)
    return blocks[0] if len(blocks) == 1 else blocks
