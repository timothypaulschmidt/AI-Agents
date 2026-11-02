from .base import Agent, WorkItem
from .helpdesk import HelpdeskAgent


def all_agents() -> dict[str, Agent]:
    """Every agent the page can run. A new agent is one more line here."""
    return {a.id: a for a in (HelpdeskAgent(),)}


__all__ = ["Agent", "WorkItem", "HelpdeskAgent", "all_agents"]
