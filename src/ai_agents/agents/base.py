"""What an agent *is*: a goal, a prompt, the tools it should have, and how to find work.

The loop that runs agents (runner.py) is generic. Everything specific to one job lives
in a subclass of Agent, so a second agent (bank fraud review, ...) is a new file here.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..mcp_client import McpTools


@dataclass(frozen=True)
class WorkItem:
    id: str
    title: str
    prompt: str  # the task message handed to the model; the details are fetched by tools


class Agent:
    id = "agent"
    name = "Agent"
    description = ""
    system_prompt = ""
    # The tools this job really needs. "Helpdesk only" tool access offers (and enforces)
    # just these; "all" gives the agent every tool the MCP server has (excessive agency).
    least_privilege_tools: frozenset[str] = frozenset()
    # Tools that only read. Dry-run lets these run and blocks everything else.
    read_tools: frozenset[str] = frozenset()

    async def next_item(self, tools: McpTools, skip: set[str]) -> WorkItem | None:
        """The next piece of work that is not in `skip`, or None if the queue is empty."""
        raise NotImplementedError

    async def start_item(self, tools: McpTools, item: WorkItem) -> None:
        """Claim the item so it is not picked up twice."""

    async def finish_item(self, tools: McpTools, item: WorkItem) -> str:
        """Called after the model is done. Make sure the item is not left half-claimed.
        Returns a short outcome for the log."""
        return "done"

    async def release_item(self, tools: McpTools, item: WorkItem) -> None:
        """Put the item back (the run was stopped, or the model could not be reached)."""
