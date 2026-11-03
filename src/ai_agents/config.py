"""Settings, read from environment variables (same image on Windows, Docker and k3s).

Two sets of endpoints exist on purpose: DIRECT (the model and the MCP server) and FAIG
(the same things reached through FortiAIGate). The route picks which are used:

  direct      model and MCP both direct
  faig_model  the model goes through FortiAIGate, MCP stays direct
  faig_all    the model and the MCP traffic both go through FortiAIGate

FAIG endpoints stay empty until FortiAIGate is configured; a route that needs an empty
one is refused. The route is held in memory, so a restart goes back to direct.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit


ROUTES = ("direct", "faig_model", "faig_all")
ROUTE_LABELS = {
    "direct": "Direct",
    "faig_model": "Model via FortiAIGate",
    "faig_all": "Model + MCP via FortiAIGate",
}


def parse_list(raw: str) -> tuple[str, ...]:
    seen: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


@dataclass(frozen=True)
class Endpoints:
    ollama_base_url: str
    gemini_base_url: str
    mcp_url: str


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    gemini_api_key: str
    gemini_base_url: str
    gemini_models: tuple[str, ...]
    mcp_url: str
    admin_token: str
    faig_ollama_base_url: str
    faig_gemini_base_url: str
    faig_mcp_url: str
    poll_seconds: float
    max_steps: int
    request_timeout: float
    allow_parallel: bool

    @property
    def faig_model_configured(self) -> bool:
        return bool(self.faig_ollama_base_url)

    @property
    def faig_all_configured(self) -> bool:
        return bool(self.faig_ollama_base_url and self.faig_mcp_url)

    @property
    def faig_configured(self) -> bool:  # kept for older callers: everything needed for faig_all
        return self.faig_all_configured

    def route_problem(self, route: str) -> str:
        """Empty when the route can be used; otherwise what is missing."""
        if route == "faig_model" and not self.faig_model_configured:
            return "FortiAIGate is not configured for the model yet (FAIG_OLLAMA_BASE_URL is empty)."
        if route == "faig_all" and not self.faig_all_configured:
            missing = [n for n, v in (("FAIG_OLLAMA_BASE_URL", self.faig_ollama_base_url),
                                      ("FAIG_MCP_SERVER_URL", self.faig_mcp_url)) if not v]
            return "FortiAIGate is not configured yet (empty: " + ", ".join(missing) + ")."
        return ""

    def endpoints(self, route: str) -> Endpoints:
        if route in ("faig_model", "faig_all"):
            mcp = self.faig_mcp_url if route == "faig_all" else self.mcp_url
            return Endpoints(self.faig_ollama_base_url, self.faig_gemini_base_url, mcp)
        return Endpoints(self.ollama_base_url, self.gemini_base_url, self.mcp_url)

    @property
    def admin_base_url(self) -> str:
        """The MCP server's own address without /mcp. Admin calls always go direct: the
        operator's reset button must keep working whatever FortiAIGate decides."""
        u = urlsplit(self.mcp_url)
        return f"{u.scheme}://{u.netloc}" if u.scheme and u.netloc else ""


def load_settings() -> Settings:
    env = os.environ.get
    return Settings(
        ollama_base_url=env("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        ollama_model=env("OLLAMA_MODEL", "qwen3:8b").strip(),
        gemini_api_key=env("GEMINI_API_KEY", "").strip(),
        gemini_base_url=env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").strip(),
        gemini_models=parse_list(env("GEMINI_MODELS", "gemini-flash-latest")),
        mcp_url=env("MCP_SERVER_URL", "http://localhost:8000/mcp").strip(),
        admin_token=env("ADMIN_TOKEN", "").strip(),
        faig_ollama_base_url=env("FAIG_OLLAMA_BASE_URL", "").strip(),
        faig_gemini_base_url=env("FAIG_GEMINI_BASE_URL", "").strip(),
        faig_mcp_url=env("FAIG_MCP_SERVER_URL", "").strip(),
        poll_seconds=max(1.0, float(env("POLL_SECONDS", "60"))),
        max_steps=max(1, int(env("MAX_STEPS", "8"))),
        request_timeout=float(env("REQUEST_TIMEOUT", "300")),
        allow_parallel=env("ALLOW_PARALLEL", "").strip().lower() in ("1", "true", "yes", "on"),
    )
