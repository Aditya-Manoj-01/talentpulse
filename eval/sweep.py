"""Chunking / retrieval sweep. Stage 1 ranks configs by gold-source hit rate (cheap, no LLM);
stage 2 (--judge-top N) runs full faithfulness judging on the best N configs.

    PYTHONPATH=src:. python eval/sweep.py --judge-top 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval.run_eval import judge_scores, load_golden, run_pipeline, summarize  # noqa: E402
from rag_service.pipeline import RagConfig, RagPipeline  # noqa: E402

GRID = {
    "chunker": ["fixed", "markdown"],
    "chunk_size": [300, 600, 1200],
    "chunk_overlap": [0, 100],
    "top_k": [2, 4, 6],
    "min_score": [0.0, 0.25],
}


def retrieval_only(pipe: RagPipeline, golden: list[dict]) -> float:
    answerable = [g for g in golden if g["source"] != "_none_"]
    hits = sum(g["source"] in {c["source"] for c in pipe.retrieve(g["question"])} for g in answerable)
    return hits / len(answerable)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--judge-top", type=int, default=0)
    args = ap.parse_args()
    golden = load_golden(ROOT / "eval" / "golden_set.jsonl")

    results = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values, strict=True))
        if params["chunk_overlap"] >= params["chunk_size"] // 2:
            continue
        cfg = RagConfig(name="sweep_" + "_".join(str(v) for v in values), embedder=args.embedder, **params)
        pipe = RagPipeline(cfg)
        results.append({"cfg": cfg, "hit_rate": retrieval_only(pipe, golden), "n_chunks": len(pipe.chunks)})
    # Prefer high recall, then fewer retrieved tokens (smaller top_k) to limit distractor context.
    results.sort(key=lambda r: (-r["hit_rate"], r["cfg"].top_k, r["cfg"].chunk_size))
    for r in results[:10]:
        c = r["cfg"]
        print(f"hit={r['hit_rate']:.3f} chunks={r['n_chunks']:3d} {c.chunker:8s} size={c.chunk_size:5d} "
              f"overlap={c.chunk_overlap:3d} k={c.top_k} min={c.min_score}")

    if args.judge_top and os.getenv("ANTHROPIC_API_KEY"):
        for r in results[: args.judge_top]:
            rows = run_pipeline(RagPipeline(r["cfg"]), golden, workers=4)
            judge_scores(rows, workers=4)
            print(json.dumps({k: v for k, v in summarize(rows, r["cfg"]).items() if k != "rag_config"}))


if __name__ == "__main__":
    main()
