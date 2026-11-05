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
from dataclasses import dataclass, field
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
    gateway: bool = False        # True when the model goes through FortiAIGate


def parse_model_map(raw: str) -> dict[str, str]:
    """"qwen3:8b=ollama-qwen3-8b,gemini-3.6-flash=google-gemini-3.6-flash" -> {plain: gateway name}."""
    out: dict[str, str] = {}
    for part in raw.split(","):
        plain, sep, wire = part.partition("=")
        if sep and plain.strip() and wire.strip():
            out[plain.strip()] = wire.strip()
    return out


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
    # Through FortiAIGate every model has the gateway's own name (provider slug + model, e.g.
    # ollama-qwen3-8b, google-gemini-3.6-flash) and the gateway wants its own key. The page and
    # Direct keep the plain names; this maps plain -> gateway name when a request goes through it.
    faig_model_map: dict = field(default_factory=dict)
    faig_api_key: str = ""
    faig_verify_tls: bool = True      # False = accept the gateway's self-signed lab certificate

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
        if route in ("faig_model", "faig_all") and not self.faig_api_key:
            return "FortiAIGate has no API key yet (FAIG_API_KEY is empty)."
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
            return Endpoints(self.faig_ollama_base_url, self.faig_gemini_base_url, mcp, gateway=True)
        return Endpoints(self.ollama_base_url, self.gemini_base_url, self.mcp_url)

    def wire_name(self, route: str, model: str) -> str:
        """The model name to put in the request: unchanged direct, the gateway's name through it."""
        if route in ("faig_model", "faig_all"):
            return self.faig_model_map.get(model, model)
        return model

    def gemini_choices(self, route: str) -> list[str]:
        """Gemini models offered on a route. Through the gateway: only mapped ones, and only when
        the gateway has a Gemini address."""
        if route in ("faig_model", "faig_all"):
            if not (self.faig_gemini_base_url and self.faig_api_key):
                return []
            return [m for m in self.gemini_models if m in self.faig_model_map]
        return list(self.gemini_models) if self.gemini_api_key else []

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
        gemini_models=parse_list(env("GEMINI_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.5-flash,gemini-robotics-er-2-preview")),
        mcp_url=env("MCP_SERVER_URL", "http://localhost:8000/mcp").strip(),
        admin_token=env("ADMIN_TOKEN", "").strip(),
        faig_ollama_base_url=env("FAIG_OLLAMA_BASE_URL", "").strip(),
        faig_gemini_base_url=env("FAIG_GEMINI_BASE_URL", "").strip(),
        faig_mcp_url=env("FAIG_MCP_SERVER_URL", "").strip(),
        faig_model_map=_model_map(env),
        faig_api_key=env("FAIG_API_KEY", "").strip(),
        faig_verify_tls=env("FAIG_VERIFY_TLS", "true").strip().lower() not in ("0", "false", "no", "off"),
        poll_seconds=max(1.0, float(env("POLL_SECONDS", "60"))),
        max_steps=max(1, int(env("MAX_STEPS", "8"))),
        request_timeout=float(env("REQUEST_TIMEOUT", "300")),
        allow_parallel=env("ALLOW_PARALLEL", "").strip().lower() in ("1", "true", "yes", "on"),
    )


def _model_map(env) -> dict[str, str]:
    return parse_model_map(env("FAIG_MODEL_MAP", ""))
