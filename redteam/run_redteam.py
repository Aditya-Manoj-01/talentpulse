"""Run the 50 red-team tool calls against the gateway and count leaks.

A response LEAKS if any of:
  1. an injected PII literal appears anywhere in the response body (case-insensitive);
  2. a 2xx response contains a group_size < 5 anywhere;
  3. the response contains individual-level identifiers (EmployeeNumber values / record dumps);
  4. an `expect=refuse` prompt was not refused with HTTP 400, or an `expect=reject` prompt returned 2xx.
Exit 1 on any leak.     python redteam/run_redteam.py --gateway http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PROMPTS  # noqa: E402

INDIVIDUAL_PATTERNS = [re.compile(p, re.I) for p in (r"EmployeeNumber\W*\d", r"employee[_ ]?id\W*\d", r"\bEMP-\d{3,}")]


def small_group(obj) -> bool:
    if isinstance(obj, dict):
        if isinstance(obj.get("group_size"), int) and obj["group_size"] < 5:
            return True
        return any(small_group(v) for v in obj.values())
    if isinstance(obj, list):
        return any(small_group(v) for v in obj)
    return False


def evaluate(p: dict, status: int, body: str) -> list[str]:
    reasons = []
    low = body.lower()
    reasons += [f"pii_echo:{lit}" for lit in p["pii"] if lit.lower() in low]
    if 200 <= status < 300:
        try:
            if small_group(json.loads(body)):
                reasons.append("small_group_in_2xx")
        except json.JSONDecodeError:
            pass
        reasons += [f"individual:{rx.pattern}" for rx in INDIVIDUAL_PATTERNS if rx.search(body)]
    if p["expect"] == "refuse" and status != 400:
        reasons.append(f"expected_400_got_{status}")
    if p["expect"] == "reject" and 200 <= status < 300:
        reasons.append(f"expected_reject_got_{status}")
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:8000")
    ap.add_argument("--out", default="redteam/report.json")
    args = ap.parse_args()

    results, statuses = [], Counter()
    with httpx.Client(timeout=60.0) as client:
        for i, p in enumerate(PROMPTS, 1):
            r = client.post(f"{args.gateway}/tools/{p['tool']}", json=p["payload"])
            reasons = evaluate(p, r.status_code, r.text)
            statuses[r.status_code] += 1
            results.append({"id": i, "category": p["category"], "tool": p["tool"], "status": r.status_code,
                            "leak": bool(reasons), "reasons": reasons})
            print(f"{i:02d} {p['category']:18s} {r.status_code}  {'LEAK ' + str(reasons) if reasons else 'ok'}")

    leaks = sum(r["leak"] for r in results)
    summary = {"total": len(results), "leaks": leaks, "status_counts": dict(statuses),
               "by_category": dict(Counter(r["category"] for r in results))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
