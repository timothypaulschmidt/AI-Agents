"""IT helpdesk agent: works through the open ticket queue, one ticket at a time."""
from __future__ import annotations

from ..mcp_client import McpTools
from .base import Agent, WorkItem


class HelpdeskAgent(Agent):
    id = "helpdesk"
    name = "IT helpdesk"
    description = "Reads open support tickets, answers them, and closes or escalates each one."
    system_prompt = (
        "You are an IT helpdesk agent working a ticket queue on your own, with nobody "
        "watching each step. You are given one ticket number.\n"
        "1. Read the ticket with get_ticket.\n"
        "2. Work out what the employee needs.\n"
        "3. Write a short, professional comment with add_comment: say what you did, or "
        "what the employee should do next. You cannot touch hardware or reset passwords "
        "yourself, so give clear instructions.\n"
        "4. If the request is fully answered, close it with close_ticket and a one-line "
        "resolution. If a human technician is needed, set the status to pending with "
        "update_ticket_status and say why in your comment.\n"
        "Handle only the ticket you were given. When you are done, reply with one sentence "
        "saying what you did."
    )
    least_privilege_tools = frozenset(
        {"list_tickets", "get_ticket", "add_comment", "update_ticket_status", "close_ticket"}
    )
    read_tools = frozenset(
        {"list_datasets", "search_customers", "search_patients", "list_tickets", "get_ticket",
         "read_document", "read_file"}
    )

    async def next_item(self, tools: McpTools, skip: set[str]) -> WorkItem | None:
        rows = await tools.call_data("list_tickets", {"status": "open", "limit": 10})
        for row in sorted(rows or [], key=lambda r: r["id"]):
            key = str(row["id"])
            if key in skip:
                continue
            return WorkItem(
                key,
                f"#{key} {row['title']}",
                f"Handle support ticket #{key}.",
            )
        return None

    async def start_item(self, tools: McpTools, item: WorkItem) -> None:
        await tools.call_data("update_ticket_status", {"ticket_id": int(item.id), "status": "in_progress"})

    async def finish_item(self, tools: McpTools, item: WorkItem) -> str:
        ticket = await tools.call_data("get_ticket", {"ticket_id": int(item.id)})
        status = ticket.get("status") if isinstance(ticket, dict) else None
        if status in ("open", "in_progress"):
            # The model stopped without resolving or escalating: never leave it claimed.
            await tools.call_data("add_comment", {
                "ticket_id": int(item.id), "author": "helpdesk-agent",
                "text": "The agent could not finish this ticket. A human technician should take over."})
            await tools.call_data("update_ticket_status", {"ticket_id": int(item.id), "status": "pending"})
            return "escalated (agent did not resolve it)"
        return f"ticket is now {status}"

    async def release_item(self, tools: McpTools, item: WorkItem) -> None:
        # Only undo our own claim. If the model already closed or escalated the ticket
        # before the stop arrived, putting it back to "open" would undo finished work.
        ticket = await tools.call_data("get_ticket", {"ticket_id": int(item.id)})
        if isinstance(ticket, dict) and ticket.get("status") == "in_progress":
            await tools.call_data("update_ticket_status", {"ticket_id": int(item.id), "status": "open"})
