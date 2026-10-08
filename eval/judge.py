"""LLM-as-judge faithfulness: decompose the answer into atomic claims, verify each against the contexts.

faithfulness = supported_claims / total_claims  (1.0 when the answer makes no factual claims, e.g. a refusal)
This mirrors RAGAS' definition so the two can be compared, and is validated against human labels (eval/kappa.py).
"""
from __future__ import annotations

import json
import os
import re

import anthropic

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")

JUDGE_PROMPT = """You are a strict evaluator of factual faithfulness for a retrieval-augmented HR assistant.

Step 1: Split the ANSWER into atomic factual claims (ignore citation markers like [1] and ignore statements that the
documents do not cover something).
Step 2: For each claim decide if it is directly supported by the CONTEXTS. A claim is supported only if the contexts
state it or it follows with no outside knowledge. Numbers, limits and conditions must match exactly.

Return ONLY JSON, no prose:
{{"claims": [{{"claim": "...", "supported": true, "evidence": "short quote or empty"}}]}}

QUESTION:
{question}

CONTEXTS:
{contexts}

ANSWER:
{answer}"""


class FaithfulnessJudge:
    def __init__(self, model: str = JUDGE_MODEL):
        self.model = model
        self.client = anthropic.Anthropic()

    def score(self, question: str, answer: str, contexts: list[str]) -> dict:
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) or "(no contexts retrieved)"
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, contexts=ctx, answer=answer)}],
        )
        raw = "".join(b.text for b in msg.content if b.type == "text")
        claims = self._parse(raw)
        total = len(claims)
        supported = sum(1 for c in claims if c.get("supported") is True)
        return {
            "faithfulness": 1.0 if total == 0 else supported / total,
            "n_claims": total,
            "n_supported": supported,
            "all_supported": supported == total,
            "claims": claims,
        }

    @staticmethod
    def _parse(raw: str) -> list[dict]:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge returned non-JSON output: {raw[:200]}")
        return json.loads(match.group(0)).get("claims", [])
