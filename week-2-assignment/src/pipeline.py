"""Orchestration. Returns the answer *and* the full trace, because the trace is
half the deliverable: query -> retrieval -> reasoning -> answer has to be visible.

Retry policy: exactly one relaxed retry. The relaxed pass drops category and
price filters but keeps the city filter and turns HyDE on. If the judge still
says insufficient, we refuse rather than answer from the model's own knowledge —
an ungrounded itinerary that reads well is the worst possible output here.
"""

from __future__ import annotations

import time

from config import FINAL_K, RERANK_INPUT, USE_HYDE
from src import generate, judge, preferences, rerank, retrieve
from src.store import Store, load


def get_store() -> Store:
    return load()


def run(query: str, store: Store | None = None, use_hyde: bool | None = None) -> dict:
    store = store or get_store()
    use_hyde = USE_HYDE if use_hyde is None else use_hyde
    started = time.perf_counter()

    trace: dict = {"query": query, "stages": []}

    # 1 — preferences ---------------------------------------------------------
    t0 = time.perf_counter()
    prefs = preferences.extract(query)
    trace["preferences"] = prefs
    trace["stages"].append({"stage": "preferences", "ms": _ms(t0), "detail": prefs.get("_source")})

    # 2 — retrieve (strict) ---------------------------------------------------
    t0 = time.perf_counter()
    candidates, rtrace = retrieve.search(
        store, query, prefs, strict=True, use_hyde=use_hyde, limit=RERANK_INPUT
    )
    trace["retrieval"] = rtrace
    trace["stages"].append({"stage": "retrieval (strict)", "ms": _ms(t0), "detail": f"{len(candidates)} candidates"})

    # 3 — rerank --------------------------------------------------------------
    t0 = time.perf_counter()
    ranked, rr = rerank.rerank(query, candidates)
    trace["rerank"] = rr
    trace["stages"].append({"stage": "rerank", "ms": _ms(t0), "detail": _rr_detail(rr)})

    # 4 — judge ---------------------------------------------------------------
    t0 = time.perf_counter()
    verdict = judge.assess(query, ranked[:FINAL_K])
    trace["judge"] = verdict
    trace["stages"].append({"stage": "judge", "ms": _ms(t0), "detail": verdict["verdict"]})

    # 5 — one relaxed retry ---------------------------------------------------
    trace["relaxed_retry"] = False
    if verdict["verdict"] == judge.INSUFFICIENT:
        trace["relaxed_retry"] = True
        t0 = time.perf_counter()
        candidates, rtrace2 = retrieve.search(
            store, query, prefs, strict=False, use_hyde=True, limit=RERANK_INPUT
        )
        ranked, rr2 = rerank.rerank(query, candidates)
        verdict2 = judge.assess(query, ranked[:FINAL_K])
        trace["retrieval_relaxed"] = rtrace2
        trace["rerank_relaxed"] = rr2
        trace["judge_relaxed"] = verdict2
        trace["stages"].append(
            {"stage": "retry (filters relaxed)", "ms": _ms(t0), "detail": verdict2["verdict"]}
        )
        verdict = verdict2

    top = ranked[:FINAL_K]
    trace["chunks"] = top
    trace["urls"] = list(dict.fromkeys(c["url"] for c in top))

    # 6 — answer or refuse ----------------------------------------------------
    t0 = time.perf_counter()
    if verdict["verdict"] == judge.INSUFFICIENT:
        result = generate.refuse(query, verdict.get("reason", "no relevant sources"))
        trace["outcome"] = "refused"
    else:
        result = generate.answer(query, top, prefs)
        trace["outcome"] = "answered"
    trace["stages"].append({"stage": "generation", "ms": _ms(t0), "detail": trace["outcome"]})

    trace["total_ms"] = int((time.perf_counter() - started) * 1000)
    return {"answer": result["answer"], "sources": result["sources"], "trace": trace}


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _rr_detail(rr: dict) -> str:
    if not rr.get("reranked"):
        return f"skipped ({rr.get('reason')})"
    return f"{rr.get('score_2plus', 0)} of {rr.get('scored', 0)} scored >= 2"
