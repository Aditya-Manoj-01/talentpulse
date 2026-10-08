"""Latency benchmark for the MCP gateway: N calls, fixed concurrency, warm-up excluded.

By default it exercises the 4 data tools (+ search_hr_policy in retrieve mode). LLM-generated answers are excluded
because their latency is dominated by the model provider, not by this system.

    python loadtest/bench.py --gateway http://localhost:8000 --calls 1000 --concurrency 8 --p95-budget-ms 120
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time

import httpx

CALLS = [
    ("get_attrition_risk", {"filters": {"department": "Sales", "overtime": "Yes"}}),
    ("get_attrition_risk", {"filters": {"job_role": "Laboratory Technician"}}),
    ("get_attrition_risk", {"filters": {"age_band": "18-30"}, "question": "Is overtime driving exits for early-career staff?"}),
    ("get_department_summary", {"department": "Research & Development"}),
    ("get_department_summary", {"department": "Sales"}),
    ("get_overtime_impact", {}),
    ("get_attrition_drivers", {"top_k": 8}),
    ("search_hr_policy", {"question": "What is the weekly overtime cap?", "mode": "retrieve"}),
]


async def run(gateway: str, n: int, concurrency: int, warmup: int, seed: int) -> dict:
    rng = random.Random(seed)
    plan = [rng.choice(CALLS) for _ in range(n)]
    sem = asyncio.Semaphore(concurrency)
    lat, errors = [], 0
    async with httpx.AsyncClient(base_url=gateway, timeout=30.0,
                                 limits=httpx.Limits(max_connections=concurrency * 2)) as client:
        for tool, payload in CALLS * max(1, warmup // len(CALLS)):
            await client.post(f"/tools/{tool}", json=payload)

        async def one(tool: str, payload: dict) -> None:
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(f"/tools/{tool}", json=payload)
                lat.append((time.perf_counter() - t0) * 1000)
                errors += r.status_code != 200

        t_start = time.perf_counter()
        await asyncio.gather(*(one(t, p) for t, p in plan))
        wall = time.perf_counter() - t_start
    q = statistics.quantiles(lat, n=100)
    return {"calls": n, "concurrency": concurrency, "errors": errors, "throughput_rps": round(n / wall, 1),
            "p50_ms": round(q[49], 1), "p95_ms": round(q[94], 1), "p99_ms": round(q[98], 1),
            "max_ms": round(max(lat), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:8000")
    ap.add_argument("--calls", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p95-budget-ms", type=float, default=120.0)
    a = ap.parse_args()
    res = asyncio.run(run(a.gateway, a.calls, a.concurrency, a.warmup, a.seed))
    res["p95_budget_ms"] = a.p95_budget_ms
    res["within_budget"] = res["p95_ms"] <= a.p95_budget_ms and res["errors"] == 0
    print(json.dumps(res, indent=2))
    return 0 if res["within_budget"] else 1


if __name__ == "__main__":
    sys.exit(main())
