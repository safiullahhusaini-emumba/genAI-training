"""LLM reranking in a single batched call.

One call scoring N candidates, not N calls scoring one candidate each: latency
matters in a UI, and the model also reranks better when it can see the
candidates side by side. Falls back to retrieval order if the call fails, so a
Groq hiccup degrades quality instead of breaking the app.
"""

from __future__ import annotations

from config import MODEL_FAST
from src.llm import chat_json

SYSTEM = """You rank travel guide passages by how well they answer a user's question.

You receive numbered passages. Score each 0-3:
  3 = directly answers the question with specific, usable detail (named places, prices, hours)
  2 = relevant and useful, but general
  1 = same city or topic, but does not help answer this question
  0 = irrelevant, navigational, or boilerplate

Return: {"scores": [{"id": <number>, "score": <0-3>, "why": "<8 words max>"}, ...]}
Score every passage you were given. Do not add passages."""


def rerank(query: str, candidates: list[dict], model: str = MODEL_FAST) -> tuple[list[dict], dict]:
    if not candidates:
        return [], {"reranked": False, "reason": "no candidates"}

    blocks = []
    for i, cand in enumerate(candidates):
        snippet = cand["text"][:600].replace("\n", " ")
        blocks.append(
            f"[{i}] (city={cand['city']}, category={cand['category']}, "
            f"price={cand['price_level']})\n{snippet}"
        )
    user = f"Question: {query}\n\nPassages:\n\n" + "\n\n".join(blocks)

    try:
        result = chat_json(SYSTEM, user, model=model, temperature=0.0, max_tokens=900)
    except Exception as exc:
        return candidates, {"reranked": False, "reason": f"{type(exc).__name__}"}

    scores = {}
    for item in result.get("scores", []):
        try:
            idx = int(item["id"])
            val = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates):
            scores[idx] = (max(0.0, min(3.0, val)), str(item.get("why", ""))[:60])

    if not scores:
        return candidates, {"reranked": False, "reason": "unparseable rerank response"}

    out = []
    for i, cand in enumerate(candidates):
        score, why = scores.get(i, (1.0, "not scored"))
        out.append({**cand, "rerank_score": score, "rerank_reason": why})

    # Retrieval score breaks ties, so fusion rank still counts within a score band.
    out.sort(key=lambda c: (c["rerank_score"], c["retrieval_score"]), reverse=True)
    return out, {
        "reranked": True,
        "scored": len(scores),
        "score_3": sum(1 for c in out if c["rerank_score"] >= 3),
        "score_2plus": sum(1 for c in out if c["rerank_score"] >= 2),
    }
