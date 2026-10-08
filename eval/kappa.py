"""Validate the LLM judge against human labels with Cohen's kappa.

1. Export a labelling sheet from an eval run:
     python eval/kappa.py export --rows eval/reports/tuned_rows.jsonl --out eval/human_labels.csv
2. A human fills the `human_faithful` column with 1 (every claim supported by the contexts) or 0.
   Label blind: hide the `judge_faithful` column while labelling.
3. Score agreement:
     python eval/kappa.py score --labels eval/human_labels.csv
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def export(rows_path: str, out: str) -> None:
    rows = pd.read_json(rows_path, lines=True)
    if "judge_all_supported" not in rows:
        sys.exit("Rows file has no judge results; run run_eval.py with --metric judge first.")
    sheet = pd.DataFrame({
        "id": rows["id"],
        "question": rows["question"],
        "answer": rows["answer"],
        "contexts": rows["contexts"].apply(lambda c: "\n---\n".join(c)),
        "judge_faithful": rows["judge_all_supported"].astype(int),
        "human_faithful": "",
    })
    sheet.to_csv(out, index=False)
    print(f"Wrote {len(sheet)} rows to {out}. Fill `human_faithful` with 0/1 (label blind to judge_faithful).")


def score(labels: str) -> float:
    df = pd.read_csv(labels)
    df = df[df["human_faithful"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    if len(df) < 20:
        sys.exit(f"Only {len(df)} labelled rows; label at least 20 (ideally all 60) for a stable kappa.")
    h, j = df["human_faithful"].astype(float).astype(int), df["judge_faithful"].astype(int)
    kappa = cohen_kappa_score(h, j)
    print(json.dumps({"n_labelled": len(df), "cohen_kappa": round(float(kappa), 3),
                      "raw_agreement": round(float((h == j).mean()), 3),
                      "confusion_matrix[human x judge]": confusion_matrix(h, j, labels=[0, 1]).tolist()}, indent=2))
    return float(kappa)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--rows", required=True)
    e.add_argument("--out", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--labels", required=True)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.rows, a.out)
    else:
        score(a.labels)
