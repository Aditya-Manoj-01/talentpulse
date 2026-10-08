"""Presidio-based PII scrubbing for every inbound string and every outbound free-text answer."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "EMPLOYEE_ID", "IP_ADDRESS", "CREDIT_CARD"]
SPACY_MODEL = os.getenv("PRESIDIO_SPACY_MODEL", "en_core_web_lg")  # Presidio default; _sm misses more names
SCORE_THRESHOLD = float(os.getenv("PRESIDIO_SCORE_THRESHOLD", "0.4"))


def _employee_id_recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity="EMPLOYEE_ID",
        patterns=[
            Pattern("emp_prefixed", r"\bEMP[-_ ]?\d{3,7}\b", 0.85),
            Pattern("employee_number_phrase", r"(?i)\b(?:employee|emp|staff)\s*(?:no\.?|number|id|#)\s*:?\s*\d{1,7}\b", 0.85),
        ],
    )


@lru_cache(maxsize=1)
def engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    nlp = NlpEngineProvider(
        nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": SPACY_MODEL}]}
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    analyzer.registry.add_recognizer(_employee_id_recognizer())
    return analyzer, AnonymizerEngine()


def scrub_text(text: str) -> tuple[str, list[str]]:
    """Returns (redacted_text, detected_entity_types)."""
    if not text or not text.strip():
        return text, []
    analyzer, anonymizer = engines()
    results = analyzer.analyze(text=text, language="en", entities=ENTITIES, score_threshold=SCORE_THRESHOLD)
    if not results:
        return text, []
    results = results + _email_derived_names(text, results)
    redacted = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
                   **{e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in ENTITIES}},
    )
    return redacted.text, sorted({r.entity_type for r in results})


def _email_derived_names(text: str, results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Deterministic backstop for NER misses: if 'priya.sharma@x.com' is present, 'Priya' / 'Sharma'
    elsewhere in the text are almost certainly the same person's name, so redact them too."""
    extra: list[RecognizerResult] = []
    for r in results:
        if r.entity_type != "EMAIL_ADDRESS":
            continue
        local = text[r.start:r.end].split("@", 1)[0]
        for token in {t for t in re.split(r"[._\-+0-9]+", local) if len(t) >= 3}:
            for m in re.finditer(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE):
                if not any(o.start <= m.start() < o.end for o in results):
                    extra.append(RecognizerResult("PERSON", m.start(), m.end(), 0.6))
    return extra


def scrub_payload(obj: Any, allowlist: frozenset[str] = frozenset()) -> tuple[Any, list[str]]:
    """Recursively scrub every string in a JSON payload. Allow-listed enum values skip NLP (latency)."""
    found: list[str] = []
    if isinstance(obj, str):
        if obj in allowlist:
            return obj, found
        return scrub_text(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k], f = scrub_payload(v, allowlist)
            found += f
        return out, found
    if isinstance(obj, list):
        out_list = []
        for v in obj:
            sv, f = scrub_payload(v, allowlist)
            out_list.append(sv)
            found += f
        return out_list, found
    return obj, found
