"""Model service: serves cohort-level attrition risk only (never per-employee scores)."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from common.cohort import CohortFilter, apply_filter
from common.data import ARTIFACTS_DIR, ID_COL, TARGET, feature_target, load_hr
from common.privacy import MIN_GROUP_SIZE, PRIVACY_REFUSAL

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    df = load_hr()
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    oof_path = ARTIFACTS_DIR / "oof_scores.csv"
    if oof_path.exists():
        scores = pd.read_csv(oof_path)
        df = df.merge(scores, on=ID_COL, how="left", validate="one_to_one")
        STATE["score_source"] = "out_of_fold"
    else:  # fallback: in-sample scores from the final model
        model = joblib.load(ARTIFACTS_DIR / "model.joblib")
        df["risk"] = model.predict_proba(feature_target(df)[0])[:, 1]
        STATE["score_source"] = "in_sample"
    STATE.update(df=df, metrics=metrics, threshold=float(metrics["decision_threshold"]))
    yield
    STATE.clear()


app = FastAPI(title="TalentPulse Model Service", version="1.0.0", lifespan=lifespan)


class CohortRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filters: CohortFilter = Field(default_factory=CohortFilter)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": STATE["metrics"]["best_model"], "scores": STATE["score_source"]}


@app.get("/model/info")
def model_info() -> dict:
    m = STATE["metrics"]
    best = m["candidates"][m["best_model"]]
    return {"best_model": m["best_model"], "cv_roc_auc_mean": best["cv_roc_auc_mean"],
            "cv_brier_mean": best["cv_brier_mean"], "positive_rate": m["positive_rate"]}


@app.post("/cohort/risk")
def cohort_risk(req: CohortRiskRequest) -> dict:
    cohort = apply_filter(STATE["df"], req.filters)
    n = len(cohort)
    if n < MIN_GROUP_SIZE:
        raise HTTPException(status_code=400, detail=PRIVACY_REFUSAL)
    risk = cohort["risk"].to_numpy()
    return {
        "cohort": req.filters.model_dump(exclude_none=True),
        "group_size": n,
        "observed_attrition_rate": round(float(cohort[TARGET].mean()), 4),
        "mean_predicted_risk": round(float(risk.mean()), 4),
        "p90_predicted_risk": round(float(np.percentile(risk, 90)), 4),
        "high_risk_share": round(float((risk >= STATE["threshold"]).mean()), 4),
        "model": STATE["metrics"]["best_model"],
        "score_source": STATE["score_source"],
    }
