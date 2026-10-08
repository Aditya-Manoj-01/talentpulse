"""RAG service: FAISS retrieval over HR policy documents with grounded, cited generation."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from common.data import REPO_ROOT
from rag_service.pipeline import RagConfig, RagPipeline

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg_path = os.getenv("RAG_CONFIG", str(REPO_ROOT / "configs" / "rag_tuned.yaml"))
    STATE["pipe"] = RagPipeline(RagConfig.from_yaml(cfg_path))
    yield
    STATE.clear()


app = FastAPI(title="TalentPulse RAG Service", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=1000)
    mode: str = Field(default="answer", pattern="^(answer|retrieve)$")


@app.get("/health")
def health() -> dict:
    p: RagPipeline = STATE["pipe"]
    return {"status": "ok", "config": p.cfg.name, "chunks": len(p.chunks), "embedder": p.cfg.embedder}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    p: RagPipeline = STATE["pipe"]
    if req.mode == "retrieve":
        return {"question": req.question, "answer": None, "contexts": p.retrieve(req.question), "config": p.cfg.name}
    return p.answer(req.question)
