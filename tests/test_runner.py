import asyncio

import pytest

from ai_agents.agents import HelpdeskAgent
from ai_agents.mcp_client import load_tools
from ai_agents.runner import AgentRunner, RunConfig
from fakes import FakeModel, Store, good_script, make_server, rogue_script, settings

TICKETS = [
    {"id": 1, "title": "VPN drops", "body": "VPN drops hourly", "status": "open"},
    {"id": 2, "title": "Printer", "body": "Printer offline", "status": "open"},
]


def build(script, store=None, **over):
    store = store or Store(TICKETS)
    server = make_server(store)
    model = FakeModel(script)

    async def factory(route):
        return await load_tools(server)

    runner = AgentRunner(HelpdeskAgent(), settings(), model.client(), tools_factory=factory)
    cfg = RunConfig(backend="ollama", model="fake", poll_seconds=30, max_steps=over.pop("max_steps", 6), **over)
    return runner, store, model, cfg


async def run_until_idle(runner, cfg, timeout=10):
    runner.start(cfg)
    for _ in range(int(timeout * 50)):
        await asyncio.sleep(0.02)
        if runner.status == "idle":
            break
    await runner.stop()


def types(runner):
    return [e["type"] for e in runner.events]


async def test_happy_path_works_every_open_ticket():
    runner, store, model, cfg = build(good_script)
    await run_until_idle(runner, cfg)
    assert {t["status"] for t in store.tickets.values()} == {"closed"}
    assert [c["author"] for c in store.tickets[1]["comments"]] == ["helpdesk-agent", "helpdesk-agent"]
    assert runner.counters["items_done"] == 2 and runner.counters["items_failed"] == 0
    assert types(runner).count("item_done") == 2
    assert runner.status == "stopped"


async def test_idles_when_queue_empty_and_wakes_on_poke():
    runner, store, model, cfg = build(good_script, Store([]))
    runner.start(cfg)
    await asyncio.sleep(0.2)
    assert runner.status == "idle"
    store.tickets[7] = dict(id=7, title="New", body="x", status="open", comments=[])
    runner.poke()
    for _ in range(500):   # up to 10 s: four model steps can be slow on a busy machine
        await asyncio.sleep(0.02)
        if store.tickets[7]["status"] == "closed":
            break
    await runner.stop()
    assert store.tickets[7]["status"] == "closed"


async def test_step_cap_escalates_instead_of_looping_forever():
    def stuck(messages, offered):
        return {"calls": [("get_ticket", {"ticket_id": 1})]}   # never finishes

    runner, store, model, cfg = build(stuck, Store(TICKETS[:1]), max_steps=3)
    await run_until_idle(runner, cfg)
    assert runner.counters["steps"] == 3
    assert store.tickets[1]["status"] == "pending"          # handed to a human
    assert any("could not finish" in c["body"] for c in store.tickets[1]["comments"])
    assert runner.counters["items_failed"] == 1


async def test_dry_run_reads_but_changes_nothing():
    runner, store, model, cfg = build(good_script, dry_run=True)
    await run_until_idle(runner, cfg)
    assert {t["status"] for t in store.tickets.values()} == {"open"}
    assert all(t["comments"] == [] for t in store.tickets.values())
    assert runner.counters["blocked"] >= 2
    assert "blocked" in types(runner)


async def test_rogue_model_with_all_tools_deletes_data():
    runner, store, model, cfg = build(rogue_script, Store(TICKETS[:1]), tool_scope="all")
    await run_until_idle(runner, cfg)
    assert store.deleted == ["banking_transactions"]


async def test_least_privilege_scope_hides_and_blocks_dangerous_tools():
    runner, store, model, cfg = build(rogue_script, Store(TICKETS[:1]), tool_scope="least")
    await run_until_idle(runner, cfg)
    assert store.deleted == []                                         # blocked at run time
    assert runner.counters["blocked"] == 1
    for req in model.requests:                                         # never even offered
        assert "delete_records" not in [t["function"]["name"] for t in req["tools"]]
    assert "delete_records" not in [e["name"] for e in runner.events if e["type"] == "tool_result" and not e["is_error"]]


async def test_stop_puts_the_claimed_ticket_back():
    gate = asyncio.Event()

    def slow(messages, offered):
        return {"calls": [("get_ticket", {"ticket_id": 1})]}

    runner, store, model, cfg = build(slow, Store(TICKETS[:1]), max_steps=50)
    runner.start(cfg)
    for _ in range(100):
        await asyncio.sleep(0.02)
        if store.tickets[1]["status"] == "in_progress":
            break
    assert store.tickets[1]["status"] == "in_progress"
    await runner.stop()
    assert store.tickets[1]["status"] == "open"
    assert runner.status == "stopped"


async def test_model_unreachable_puts_ticket_back_and_keeps_running():
    import httpx
    store = Store(TICKETS[:1])
    server = make_server(store)

    def boom(request):
        raise httpx.ConnectError("down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom))

    async def factory(route):
        return await load_tools(server)

    runner = AgentRunner(HelpdeskAgent(), settings(), client, tools_factory=factory)
    runner.start(RunConfig(model="fake", poll_seconds=30))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if runner.status == "idle":
            break
    assert store.tickets[1]["status"] == "open"
    assert any(e["type"] == "error" for e in runner.events)
    assert runner.running
    await runner.stop()


async def test_subscribers_receive_live_events():
    runner, store, model, cfg = build(good_script, Store(TICKETS[:1]))
    q = runner.subscribe()
    await run_until_idle(runner, cfg)
    got = []
    while not q.empty():
        got.append(q.get_nowait()["type"])
    assert "item_start" in got and "tool_call" in got and "item_done" in got


async def test_releasing_never_reopens_a_ticket_that_is_already_finished():
    """Stop can arrive just after the model closed the ticket: that work must stay done."""
    store = Store(TICKETS[:2])
    server = make_server(store)
    tools = await load_tools(server)
    agent = HelpdeskAgent()
    from ai_agents.agents import WorkItem
    store.tickets[1]["status"] = "closed"
    store.tickets[2]["status"] = "pending"
    await agent.release_item(tools, WorkItem("1", "t", "p"))
    await agent.release_item(tools, WorkItem("2", "t", "p"))
    assert (store.tickets[1]["status"], store.tickets[2]["status"]) == ("closed", "pending")
    store.tickets[1]["status"] = "in_progress"                 # a genuinely half-done claim
    await agent.release_item(tools, WorkItem("1", "t", "p"))
    assert store.tickets[1]["status"] == "open"
