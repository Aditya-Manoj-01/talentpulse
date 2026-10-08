"""Analytics service: department summaries, overtime causal results, global drivers. Aggregates only."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from common.cohort import Department
from common.data import ARTIFACTS_DIR, TARGET, load_hr
from common.privacy import MIN_GROUP_SIZE, PRIVACY_REFUSAL

STATE: dict = {}


def _read_json(name: str) -> dict | None:
    path = ARTIFACTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["df"] = load_hr()
    STATE["causal"] = _read_json("causal_results.json")
    STATE["shap"] = _read_json("shap_global_importance.json")
    STATE["fairness"] = _read_json("fairness_report.json")
    yield
    STATE.clear()


app = FastAPI(title="TalentPulse Analytics Service", version="1.0.0", lifespan=lifespan)


class DepartmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    department: Department


def _summ(g: pd.DataFrame) -> dict:
    return {
        "group_size": int(len(g)),
        "attrition_rate": round(float(g[TARGET].mean()), 4),
        "overtime_rate": round(float((g["OverTime"] == "Yes").mean()), 4),
        "median_years_at_company": float(g["YearsAtCompany"].median()),
        "mean_job_satisfaction": round(float(g["JobSatisfaction"].mean()), 2),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "causal_loaded": STATE["causal"] is not None, "shap_loaded": STATE["shap"] is not None}


@app.post("/department/summary")
def department_summary(req: DepartmentRequest) -> dict:
    dept = STATE["df"][STATE["df"]["Department"] == req.department]
    if len(dept) < MIN_GROUP_SIZE:
        raise HTTPException(status_code=400, detail=PRIVACY_REFUSAL)
    by_role, suppressed = {}, 0
    for role, g in dept.groupby("JobRole"):
        if len(g) < MIN_GROUP_SIZE:
            suppressed += 1  # small-cell suppression: the cell is omitted, not zero-filled
            continue
        by_role[role] = _summ(g)
    return {"department": req.department, **_summ(dept), "by_job_role": by_role,
            "suppressed_cells": suppressed}


@app.get("/overtime/impact")
def overtime_impact() -> dict:
    if STATE["causal"] is None:
        raise HTTPException(status_code=503, detail="Run notebooks/01_causal_and_stats.ipynb first")
    return {"n_records": STATE["causal"]["n_records"], **STATE["causal"]["overtime"],
            "top_driver": STATE["causal"]["top_driver"]}


@app.get("/drivers")
def drivers(top_k: int = Query(default=8, ge=1, le=20)) -> dict:
    if STATE["shap"] is None:
        raise HTTPException(status_code=503, detail="Run model_service.train first")
    items = list(STATE["shap"].items())[:top_k]
    fairness = STATE["fairness"] or {}
    return {
        "method": "mean |SHAP| aggregated to source feature (global, cohort-level)",
        "drivers": [{"feature": k, "mean_abs_shap": v} for k, v in items],
        "fairness_disparities": fairness.get("disparities"),
    }
