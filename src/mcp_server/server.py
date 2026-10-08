"""TalentPulse MCP gateway.

Exposes 5 aggregate-only tools over HTTP (`/tools/<name>`) plus an MCP-style manifest (`/tools`).
`mcp_bridge.py` wraps these endpoints in the official MCP protocol (stdio or streamable HTTP) so any MCP client
(Claude Desktop, Claude Code, Cursor, ...) can call them.

Guardrails, applied to EVERY tool call in this order:
  1. PII scrubbing   - Presidio redacts names/emails/phones/employee IDs from the raw JSON body *before* parsing.
  2. Schema lock     - only enumerated cohort attributes are accepted; unknown fields are rejected (extra=forbid).
  3. Aggregate-only  - requested `group_size` < 5 -> HTTP 400; resolved cohort size < 5 -> HTTP 400.
  4. Output check    - downstream responses are re-checked for small cells and free text is re-scrubbed.
Validation errors never echo user input back (prevents reflecting PII in error messages).
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common.cohort import CohortFilter, Department
from common.data import DEPARTMENTS, JOB_ROLES
from common.privacy import MIN_GROUP_SIZE, PRIVACY_REFUSAL
from mcp_server.pii import engines, scrub_payload, scrub_text

log = logging.getLogger("mcp_gateway")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")
ANALYTICS_URL = os.getenv("ANALYTICS_SERVICE_URL", "http://localhost:8002")
RAG_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8003")

ENUM_ALLOWLIST = frozenset(
    DEPARTMENTS + JOB_ROLES + ["Yes", "No", "18-30", "31-40", "41-50", "51+",
                               "Non-Travel", "Travel_Rarely", "Travel_Frequently", "retrieve", "answer"]
)


# --------------------------------------------------------------------------- request schemas
class AttritionRiskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filters: CohortFilter = Field(default_factory=CohortFilter)
    group_size: int = Field(default=MIN_GROUP_SIZE, description="Minimum aggregate size requested (>= 5).")
    question: str | None = Field(default=None, max_length=1000, description="Optional analyst context (PII-scrubbed).")


class DepartmentSummaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    department: Department
    group_size: int = MIN_GROUP_SIZE


class OvertimeImpactArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DriversArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(default=8, ge=1, le=20)


class PolicySearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=1000)
    mode: str = Field(default="answer", pattern="^(answer|retrieve)$")


TOOLS: dict[str, dict[str, Any]] = {
    "get_attrition_risk": {"schema": AttritionRiskArgs,
                           "description": "Mean predicted attrition risk and observed attrition for a cohort (n >= 5)."},
    "get_department_summary": {"schema": DepartmentSummaryArgs,
                               "description": "Department headcount, attrition, overtime, job-role mix; cells < 5 suppressed."},
    "get_overtime_impact": {"schema": OvertimeImpactArgs,
                            "description": "Chi-square, odds ratios and propensity-matched effect of overtime on attrition."},
    "get_attrition_drivers": {"schema": DriversArgs,
                              "description": "Global attrition drivers (mean |SHAP|) and Fairlearn disparity summary."},
    "search_hr_policy": {"schema": PolicySearchArgs,
                         "description": "Answer HR-policy questions grounded in the policy corpus (RAG) with citations."},
}


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    engines()  # warm Presidio + spaCy once at startup (keeps p95 latency low)
    app.state.http = getattr(app.state, "http", None) or httpx.AsyncClient(timeout=30.0)
    yield
    await app.state.http.aclose()


app = FastAPI(title="TalentPulse MCP Gateway", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Report WHERE validation failed, never WHAT was sent (the input may contain PII).
    fields = sorted({".".join(str(p) for p in e["loc"] if p != "body") for e in exc.errors()})
    return JSONResponse(status_code=422, content={"detail": "Invalid arguments", "fields": fields})


def _refuse() -> None:
    raise HTTPException(status_code=400, detail=PRIVACY_REFUSAL)


async def _parse(request: Request, schema: type[BaseModel]) -> tuple[BaseModel, list[str]]:
    """Guardrail 1 + 2: scrub the raw JSON body with Presidio, then validate against a locked schema."""
    raw = await request.body()
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Body must be valid JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Body must be a JSON object")
    scrubbed, entities = scrub_payload(payload, ENUM_ALLOWLIST)
    try:
        args = schema.model_validate(scrubbed)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        raise HTTPException(status_code=422, detail={"message": "Invalid arguments", "fields": fields}) from None
    if entities:
        log.info("pii_redacted entities=%s", entities)  # log entity TYPES only, never values
    if getattr(args, "group_size", MIN_GROUP_SIZE) < MIN_GROUP_SIZE:  # Guardrail 3a
        _refuse()
    return args, entities


async def _downstream(request: Request, method: str, url: str, **kw) -> dict:
    resp = await request.app.state.http.request(method, url, **kw)
    if resp.status_code == 400 and PRIVACY_REFUSAL in resp.text:
        _refuse()
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Upstream error {resp.status_code}")
    return resp.json()


def _assert_aggregate(obj: Any, floor: int) -> None:
    """Guardrail 4: any group_size below the floor anywhere in a response blocks the whole response."""
    if isinstance(obj, dict):
        if "group_size" in obj and isinstance(obj["group_size"], int) and obj["group_size"] < floor:
            _refuse()
        for v in obj.values():
            _assert_aggregate(v, floor)
    elif isinstance(obj, list):
        for v in obj:
            _assert_aggregate(v, floor)


def _envelope(tool: str, data: dict, redacted: list[str], t0: float) -> dict:
    return {"tool": tool, "data": data, "privacy": {"min_group_size": MIN_GROUP_SIZE,
                                                    "pii_entities_redacted": sorted(set(redacted))},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


# --------------------------------------------------------------------------- endpoints
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/tools")
async def list_tools() -> dict:
    """MCP-style tool manifest (name, description, JSON Schema for inputs)."""
    return {"tools": [{"name": n, "description": t["description"], "inputSchema": t["schema"].model_json_schema()}
                      for n, t in TOOLS.items()]}


@app.post("/tools/get_attrition_risk")
async def get_attrition_risk(request: Request) -> dict:
    t0 = time.perf_counter()
    args, redacted = await _parse(request, AttritionRiskArgs)
    floor = max(MIN_GROUP_SIZE, args.group_size)
    data = await _downstream(request, "POST", f"{MODEL_URL}/cohort/risk",
                             json={"filters": args.filters.model_dump(exclude_none=True)})
    if data.get("group_size", 0) < floor:
        _refuse()
    _assert_aggregate(data, floor)
    return _envelope("get_attrition_risk", data, redacted, t0)


@app.post("/tools/get_department_summary")
async def get_department_summary(request: Request) -> dict:
    t0 = time.perf_counter()
    args, redacted = await _parse(request, DepartmentSummaryArgs)
    floor = max(MIN_GROUP_SIZE, args.group_size)
    data = await _downstream(request, "POST", f"{ANALYTICS_URL}/department/summary",
                             json={"department": args.department})
    # Re-apply the caller's (possibly stricter) floor to job-role cells instead of failing the whole call.
    roles = data.get("by_job_role", {})
    kept = {r: v for r, v in roles.items() if v["group_size"] >= floor}
    data["suppressed_cells"] = data.get("suppressed_cells", 0) + (len(roles) - len(kept))
    data["by_job_role"] = kept
    _assert_aggregate(data, floor)
    return _envelope("get_department_summary", data, redacted, t0)


@app.post("/tools/get_overtime_impact")
async def get_overtime_impact(request: Request) -> dict:
    t0 = time.perf_counter()
    _, redacted = await _parse(request, OvertimeImpactArgs)
    data = await _downstream(request, "GET", f"{ANALYTICS_URL}/overtime/impact")
    return _envelope("get_overtime_impact", data, redacted, t0)


@app.post("/tools/get_attrition_drivers")
async def get_attrition_drivers(request: Request) -> dict:
    t0 = time.perf_counter()
    args, redacted = await _parse(request, DriversArgs)
    data = await _downstream(request, "GET", f"{ANALYTICS_URL}/drivers", params={"top_k": args.top_k})
    return _envelope("get_attrition_drivers", data, redacted, t0)


@app.post("/tools/search_hr_policy")
async def search_hr_policy(request: Request) -> dict:
    t0 = time.perf_counter()
    args, redacted = await _parse(request, PolicySearchArgs)
    data = await _downstream(request, "POST", f"{RAG_URL}/query",
                             json={"question": args.question, "mode": args.mode})
    if data.get("answer"):  # outbound scrub: the LLM must never surface PII it inferred or echoed
        data["answer"], out_entities = scrub_text(data["answer"])
        redacted += out_entities
    return _envelope("search_hr_policy", data, redacted, t0)
