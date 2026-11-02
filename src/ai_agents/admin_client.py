"""The operator's channel to the MCP server's /admin routes (reset, insert, stats).

Always goes straight to the MCP server, never through FortiAIGate: the reset button
must keep working whatever the firewall decides, and it is not something an agent does.
"""
from __future__ import annotations

import httpx

from .config import Settings


class AdminError(Exception):
    pass


class AdminClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        s = self.settings
        if not s.admin_token:
            raise AdminError("ADMIN_TOKEN is not set, so the reset/insert controls are off.")
        if not s.admin_base_url:
            raise AdminError("MCP_SERVER_URL is not a full URL.")
        try:
            resp = await self.client.request(
                method, s.admin_base_url + path, json=body,
                headers={"Authorization": f"Bearer {s.admin_token}"}, timeout=httpx.Timeout(10.0))
        except httpx.HTTPError as exc:
            raise AdminError(f"MCP server unreachable: {type(exc).__name__}: {exc}") from exc
        if resp.status_code == 401:
            raise AdminError("The MCP server rejected the admin token.")
        if resp.status_code == 404:
            raise AdminError("The MCP server has no admin routes (it was started without ADMIN_TOKEN).")
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            raise AdminError(str(data.get("error", f"HTTP {resp.status_code}")))
        return data

    async def stats(self) -> dict:
        return await self._request("GET", "/admin/stats")

    async def reset(self, scenario: str, full: bool = False) -> dict:
        return await self._request("POST", "/admin/reset", {"scenario": scenario, "full": full})

    async def insert(self, kind: str, count: int) -> dict:
        return await self._request("POST", "/admin/insert", {"kind": kind, "count": count})
