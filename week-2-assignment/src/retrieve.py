"""Hybrid retrieval: dense + sparse (BM25), fused with RRF, filtered server-side.

Why hybrid and not dense alone: travel queries are full of proper nouns —
"Pergamonmuseum", "Markthalle Neun", "Tempelhofer Feld". A 384-dim embedding
smears those into a generic "museum in Berlin" region of the space, while the
sparse side matches them exactly. Dense handles the fuzzy half of the query
("somewhere atmospheric for a rainy afternoon"), sparse handles the precise
half.

The filter is built once and handed to *both* prefetches, never to the outer
query — fusion only ever sees prefetch output, so a filter on the outer query
would run after fusion, discarding relevant candidates that fusion never got
a chance to surface. City is the only condition that survives the relaxed
retry: wrong-city advice is worse than none, so it's never dropped.

HyDE is optional and off the critical path: it costs one extra fast call and
mainly helps vague queries. There is a toggle in the UI so you can see the
retrieved set change with and without it. The sparse side always uses the
real query — searching lexically over invented text is a good way to
retrieve invented places.
"""

from __future__ import annotations

from qdrant_client import models

from config import BM25_K, DENSE_K, DENSE_VECTOR, MODEL_FAST, RRF_K, SPARSE_VECTOR
from src.embedder import embed_query
from src.llm import chat
from src.store import PAYLOAD_FIELDS, Store, sparse_query

HYDE_SYSTEM = (
    "You write a short, plausible paragraph from a travel guide that would "
    "answer the user's question. Invented specifics are fine — this text is "
    "used only as a search probe, never shown to anyone. 80 words maximum, "
    "plain prose, no preamble."
)


def hyde(query: str, model: str = MODEL_FAST) -> str | None:
    """Generate a hypothetical passage and search with that instead of the query."""
    try:
        doc = chat(HYDE_SYSTEM, query, model=model, temperature=0.5, max_tokens=180)
        return doc or None
    except Exception:
        return None


def _build_filter(prefs: dict, strict: bool) -> models.Filter | None:
    """Build the payload filter passed to both the dense and sparse prefetches."""
    must: list[models.Condition] = []

    city = prefs.get("city")
    if city:
        must.append(
            models.FieldCondition(
                key="city_key",
                match=models.MatchValue(value=str(city).strip().lower()),
            )
        )

    if strict:
        cats = prefs.get("categories") or []
        if cats:
            must.append(models.FieldCondition(key="category", match=models.MatchAny(any=list(cats))))
        price = prefs.get("price_level")
        if price:
            must.append(models.FieldCondition(key="price_level", match=models.MatchValue(value=price)))

    # None rather than Filter(must=[]): an empty `must` is legal but runs the
    # filtering machinery for nothing.
    return models.Filter(must=must) if must else None


def _filter_summary(prefs: dict, strict: bool) -> str | None:
    parts = []
    if prefs.get("city"):
        parts.append(f"city={prefs['city']}")
    if strict:
        if prefs.get("categories"):
            parts.append(f"categories={','.join(prefs['categories'])}")
        if prefs.get("price_level"):
            parts.append(f"price={prefs['price_level']}")
    return ", ".join(parts) if parts else None


def search(
    store: Store,
    query: str,
    prefs: dict,
    *,
    strict: bool = True,
    use_hyde: bool = True,
    limit: int = 12,
) -> tuple[list[dict], dict]:
    """Returns (candidates, trace). Candidates carry a fused score and their metadata."""
    trace: dict = {"strict_filters": strict, "hyde_used": False}

    probe = query
    if use_hyde:
        doc = hyde(query)
        if doc:
            probe = f"{query}\n{doc}"
            trace["hyde_used"] = True
            trace["hyde_doc"] = doc

    qfilter = _build_filter(prefs, strict)
    trace["filter_summary"] = _filter_summary(prefs, strict)
    trace["qdrant_filter"] = qfilter.model_dump(exclude_none=True) if qfilter else None

    dense_vec = embed_query(probe)[0].tolist()
    sparse_vec = sparse_query(query)  # sparse side always uses the real query, not the HyDE probe

    prefetch = [
        models.Prefetch(query=dense_vec, using=DENSE_VECTOR, filter=qfilter, limit=DENSE_K),
        models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR, filter=qfilter, limit=BM25_K),
    ]

    # Three requests, one round trip. Server-side fusion returns only the final
    # points, so the dense-only and sparse-only branches exist purely to keep
    # the trace honest — pipeline.py calls the trace "half the deliverable",
    # and app.py's debug panel headlines these per-branch counts.
    requests = [
        models.QueryRequest(query=dense_vec, using=DENSE_VECTOR, filter=qfilter, limit=DENSE_K, with_payload=False),
        models.QueryRequest(query=sparse_vec, using=SPARSE_VECTOR, filter=qfilter, limit=BM25_K, with_payload=False),
        models.QueryRequest(
            prefetch=prefetch,
            query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
            limit=limit,
            with_payload=PAYLOAD_FIELDS,
        ),
    ]
    dense_res, sparse_res, fused_res = store.client.query_batch_points(
        collection_name=store.collection, requests=requests
    )

    dense_ids = {p.id for p in dense_res.points}
    sparse_ids = {p.id for p in sparse_res.points}
    trace["dense_hits"] = len(dense_ids)
    trace["bm25_hits"] = len(sparse_ids)
    trace["fused_pool"] = len(dense_ids | sparse_ids)

    candidates = [
        {**point.payload, "retrieval_score": round(point.score, 5), "point_id": str(point.id)}
        for point in fused_res.points
    ]

    trace["after_filters"] = len(candidates)
    return candidates, trace
