# AI Agents

Agents that work on their own, in a loop, using the tools of the [MCP server](https://github.com/timothypaulschmidt/Ai-Assistant-MCP-Server). Built to learn AI security: first watch an agent do its job, then watch it go rogue, then stop it.

First agent: **IT helpdesk**. It polls the ticket queue, reads each ticket, comments, and closes or escalates it. More agents (bank fraud review, ...) plug in as one file each under `src/ai_agents/agents/`.

## The page

One page controls everything: start/stop, a live action log, ticket counts, and buttons that reset the data or add tickets while the agent runs.

| Control | What it does |
|---|---|
| Model source / Model | Local Ollama model (e.g. `qwen3:8b`) or Gemini if a key is set |
| Tool access | **All MCP tools** = excessive access (the rogue demo). **Helpdesk tools only** = least privilege: other tools are not offered *and* are refused if the model asks for them |
| Check queue every | The idle wait between looks at the queue (the slow loop) |
| Max steps per ticket | Hard cap on model rounds; a ticket the agent can't finish goes to `pending` for a human |
| Dry run | Read tools run; writes are logged as BLOCKED and not done |
| Reset / + tickets | Calls the MCP server's token-protected `/admin` routes (not MCP tools, so no model can use them) |
| Traffic: Direct / Via FortiAIGate | Switch for the model and MCP traffic. Disabled until the `FAIG_*` settings exist |

The damage line under the counters watches `banking_transactions` (1500 when healthy) and emails sent, so a rogue action shows even when every ticket looks closed.

## Run locally (Windows, step by step)

You need the MCP server running with an admin token, and Ollama (or a Gemini key).

```powershell
# 1. MCP server (its own folder, own terminal)
$env:ADMIN_TOKEN = "lab-token"; $env:MCP_MODE = "vulnerable"
python -m ai_assistant_mcp_server --transport streamable-http --port 8000

# 2. This repo (another terminal)
cd <your clone of AI-Agents>
pip install -r requirements-dev.txt
python -m pytest -q
$env:MCP_SERVER_URL = "http://localhost:8000/mcp"
$env:ADMIN_TOKEN = "lab-token"
$env:OLLAMA_BASE_URL = "http://k3s-qa.fortinetlabs.com:11434"   # or your local Ollama
$env:POLL_SECONDS = "10"
python -m uvicorn ai_agents.web.main:app --app-dir src --port 8090
```

Open http://localhost:8090. Port 8090 avoids the Assistant (8080) and MCP server (8000).

## The demo

1. **Works:** Reset to *10 normal tickets*, Start. The queue drains.
2. **Goes rogue:** Reset to *10 tickets, one poisoned* (tick *Also rebuild every table*), tool access *All MCP tools*, Start. Every ticket closes, but `banking_transactions` drops to 0 and an email is "sent".
3. **Contained by design:** same data, tool access *Helpdesk tools only*. The poisoned ticket still reaches the model; the dangerous call is refused.
4. **Contained by FortiAIGate:** later, with the Traffic switch on.

Whether a model obeys the poisoned ticket depends on the model: small local models (qwen3:8b) tend to, Gemini often doesn't.

## Deploy to k3s

```powershell
docker build -t timothypaulschmidt/ai-agents:v0.1.0 .
docker push timothypaulschmidt/ai-agents:v0.1.0
ssh administrator@k3s-qa.fortinetlabs.com "mkdir -p ~/deploy"
scp -r deploy administrator@k3s-qa.fortinetlabs.com:~/deploy/ai-agents
```

On the VM (the MCP server's `mcp-admin` Secret must exist; see that repo's `deploy/README.md`):

```bash
kubectl apply -k ~/deploy/ai-agents/k8s
kubectl -n ai-assistant get pods -l app.kubernetes.io/name=ai-agents
kubectl -n ai-assistant logs -f deploy/ai-agents      # the same action log as the page
```

Page URLs (both are NodePort 30090, plain HTTP for now):

| Environment | URL |
|---|---|
| QA | http://agents-qa.fortinetlabs.com:30090 |
| Production | http://agents.fortinetlabs.com:30090 |

The page has no login and can reset data and run agents. In QA that is fine on the lab network. In **production (AWS)** limit port 30090 in the security group to your own IP, or use an SSH tunnel, until a login is added.

## Design notes

- `runner.py` is the generic loop; `agents/` holds one job per file (prompt, least-privilege tools, how to find and claim work).
- The model only *asks* for tools; the runner decides whether to run them. Tool scope, dry run, the step cap and stop all live there.
- All model and MCP traffic goes through `Settings.endpoints(route)`. FortiAIGate later means filling in the `FAIG_OLLAMA_BASE_URL`, `FAIG_GEMINI_BASE_URL` and `FAIG_MCP_SERVER_URL` variables; the switch changes which set is used, even while an agent runs (it applies from the next ticket).
- `providers.py` and `mcp_client.py` are copied from the AI Assistant. When a third app needs them they should move into a shared package.
