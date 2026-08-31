"""Lightweight gate: is this context good enough to answer on?

The reranker already produced per-chunk relevance scores, so the cheap check is
just counting them. The LLM only gets called when that count is ambiguous —
enough material to be arguable, not enough to be obvious. This keeps the gate
at roughly zero extra calls on good queries and one call on borderline ones.
"""

from __future__ import annotations

from config import JUDGE_PASS_THRESHOLD, MODEL_FAST
from src.llm import chat_json

SYSTEM = """You decide whether a set of retrieved passages is sufficient to answer
a travel question without guessing.

Return: {"verdict": "context_good" | "context_insufficient", "reason": "<15 words max>", "missing": "<what is absent, or null>"}

Say context_insufficient when the passages are about the wrong city, cover the
topic only in passing, or would force the answer to invent venues, prices or hours.

Judge the raw material, not the finished answer. Passages that name enough real
places to build from are sufficient — do not require them to already contain an
itinerary, a schedule, or transport directions."""

GOOD = "context_good"
INSUFFICIENT = "context_insufficient"


def assess(query: str, chunks: list[dict], model: str = MODEL_FAST) -> dict:
    if not chunks:
        return {"verdict": INSUFFICIENT, "reason": "Nothing retrieved.", "via": "count"}

    strong = [c for c in chunks if c.get("rerank_score", 0) >= 2]

    if len(strong) >= JUDGE_PASS_THRESHOLD:
        return {
            "verdict": GOOD,
            "reason": f"{len(strong)} passages scored relevant or better.",
            "via": "count",
        }
    if len(strong) == 0:
        return {
            "verdict": INSUFFICIENT,
            "reason": "No passage scored above marginal relevance.",
            "via": "count",
        }

    # Borderline: 1-2 strong passages. Ask the model.
    blocks = "\n\n".join(
        f"[{i}] {c['text'][:450]}" for i, c in enumerate(chunks[:6])
    )
    try:
        result = chat_json(
            SYSTEM,
            f"Question: {query}\n\nPassages:\n\n{blocks}",
            model=model,
            temperature=0.0,
            max_tokens=200,
        )
    except Exception:
        return {"verdict": GOOD, "reason": "Judge unavailable; proceeding.", "via": "fallback"}

    verdict = result.get("verdict")
    if verdict not in (GOOD, INSUFFICIENT):
        verdict = GOOD
    return {
        "verdict": verdict,
        "reason": str(result.get("reason", ""))[:120],
        "missing": result.get("missing"),
        "via": "llm",
    }
