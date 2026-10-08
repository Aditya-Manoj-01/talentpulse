"""RAG evaluation harness with a CI quality gate.

    PYTHONPATH=src:. python eval/run_eval.py --config configs/rag_tuned.yaml --metric judge --threshold 0.90

Metrics
  faithfulness_judge  - LLM-as-judge (eval/judge.py), validated against human labels via Cohen's kappa
  faithfulness_ragas  - RAGAS Faithfulness (optional, --metric ragas|both)
  source_hit_rate     - fraction of answerable questions whose gold source document was retrieved (no LLM)
  refusal_accuracy    - fraction of unanswerable questions the system correctly declined
Exit code 1 if the gate metric is below --threshold (fails the GitHub Actions check).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from rag_service.pipeline import RagConfig, RagPipeline  # noqa: E402

REFUSAL = "do not cover"


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_pipeline(pipe: RagPipeline, golden: list[dict], workers: int) -> list[dict]:
    def one(item: dict) -> dict:
        res = pipe.answer(item["question"])
        return {**item, "answer": res["answer"], "contexts": [c["text"] for c in res["contexts"]],
                "retrieved_sources": [c["source"] for c in res["contexts"]]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, golden))


def judge_scores(rows: list[dict], workers: int) -> None:
    from eval.judge import FaithfulnessJudge

    judge = FaithfulnessJudge()

    def one(r: dict) -> dict:
        return judge.score(r["question"], r["answer"], r["contexts"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r, s in zip(rows, ex.map(one, rows), strict=True):
            r["faithfulness_judge"] = s["faithfulness"]
            r["judge_all_supported"] = s["all_supported"]
            r["judge_claims"] = s["claims"]


def ragas_scores(rows: list[dict]) -> None:
    """RAGAS 0.2.x API (pinned in requirements-eval.txt)."""
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "retrieved_contexts": r["contexts"] or [""], "response": r["answer"],
         "reference": r["ground_truth"]} for r in rows
    ])
    llm = LangchainLLMWrapper(ChatAnthropic(model=os.getenv("JUDGE_MODEL", "claude-sonnet-5-5"), temperature=0))
    result = evaluate(ds, metrics=[Faithfulness()], llm=llm, show_progress=False)
    for r, v in zip(rows, result.to_pandas()["faithfulness"].tolist(), strict=True):
        r["faithfulness_ragas"] = float(v) if pd.notna(v) else None


def summarize(rows: list[dict], cfg: RagConfig) -> dict:
    df = pd.DataFrame(rows)
    answerable = df[df["source"] != "_none_"]
    unanswerable = df[df["source"] == "_none_"]
    summary = {
        "config": cfg.name,
        "n_questions": len(df),
        "source_hit_rate": round(float(answerable.apply(lambda r: r["source"] in r["retrieved_sources"], axis=1).mean()), 4),
        "refusal_accuracy": round(float(unanswerable["answer"].str.contains(REFUSAL, case=False).mean()), 4)
        if len(unanswerable) else None,
        "rag_config": cfg.__dict__,
    }
    for col in ("faithfulness_judge", "faithfulness_ragas"):
        if col in df and df[col].notna().any():
            summary[col] = round(float(df[col].dropna().mean()), 4)
    return summary


def write_step_summary(summary: dict, gate: str, threshold: float, passed: bool) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = [f"## RAG eval — `{summary['config']}` {'✅' if passed else '❌'}", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items() if k != "rag_config"]
    lines += ["", f"Gate: `{gate}` ≥ {threshold}"]
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "rag_tuned.yaml"))
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden_set.jsonl"))
    ap.add_argument("--metric", choices=["judge", "ragas", "both", "none"], default="judge")
    ap.add_argument("--gate", default="faithfulness_judge")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--embedder", default=None, help="Override embedder (e.g. tfidf-lsa)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    args = ap.parse_args()

    cfg = RagConfig.from_yaml(args.config)
    if args.embedder:
        cfg.embedder = args.embedder
    if args.metric != "none" and not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is required for generation and judging.", file=sys.stderr)
        return 2

    golden = load_golden(Path(args.golden))
    rows = run_pipeline(RagPipeline(cfg), golden, args.workers)
    if args.metric in ("judge", "both"):
        judge_scores(rows, args.workers)
    if args.metric in ("ragas", "both"):
        ragas_scores(rows)

    summary = summarize(rows, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg.name}_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_json(out / f"{cfg.name}_rows.jsonl", orient="records", lines=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "rag_config"}, indent=2))

    if args.metric == "none":
        return 0
    value = summary.get(args.gate)
    passed = value is not None and value >= args.threshold
    write_step_summary(summary, args.gate, args.threshold, passed)
    print(f"GATE {'PASSED' if passed else 'FAILED'}: {args.gate}={value} (threshold {args.threshold})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
