"""Official MCP protocol bridge for the TalentPulse gateway.

    # stdio (Claude Desktop / Claude Code):
    GATEWAY_URL=http://localhost:8000 python -m mcp_server.mcp_bridge
    # streamable HTTP:
    python -m mcp_server.mcp_bridge --transport streamable-http --port 8010

Every tool forwards to the guarded REST gateway, so guardrails cannot be bypassed via MCP.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer  # mcp>=2.0 (FastMCP was renamed MCPServer)

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
mcp = MCPServer(name="talentpulse", instructions="Aggregate-only people analytics. Cohorts < 5 are refused.")


async def _call(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{GATEWAY_URL}/tools/{tool}", json=payload)
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        return {"error": detail, "status": r.status_code}
    return r.json()


@mcp.tool()
async def get_attrition_risk(department: str | None = None, job_role: str | None = None,
                             overtime: str | None = None, age_band: str | None = None,
                             job_level: int | None = None, group_size: int = 5) -> dict:
    """Cohort-level predicted and observed attrition. Refuses cohorts smaller than 5 employees."""
    filters = {k: v for k, v in dict(department=department, job_role=job_role, overtime=overtime,
                                     age_band=age_band, job_level=job_level).items() if v is not None}
    return await _call("get_attrition_risk", {"filters": filters, "group_size": group_size})


@mcp.tool()
async def get_department_summary(department: str, group_size: int = 5) -> dict:
    """Department headcount, attrition, overtime rate and job-role breakdown (small cells suppressed)."""
    return await _call("get_department_summary", {"department": department, "group_size": group_size})


@mcp.tool()
async def get_overtime_impact() -> dict:
    """Statistical and propensity-matched causal estimate of overtime's effect on attrition."""
    return await _call("get_overtime_impact", {})


@mcp.tool()
async def get_attrition_drivers(top_k: int = 8) -> dict:
    """Global attrition drivers from SHAP plus Fairlearn disparity metrics."""
    return await _call("get_attrition_drivers", {"top_k": top_k})


@mcp.tool()
async def search_hr_policy(question: str) -> dict:
    """Grounded answers about HR policies (overtime, leave, remote work, compensation, privacy)."""
    return await _call("search_hr_policy", {"question": question})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    if a.transport == "streamable-http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=a.port)
    else:
        mcp.run(transport="stdio")
