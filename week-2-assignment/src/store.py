"""Qdrant for both dense and sparse search, one collection, filters applied
server-side during retrieval — not as a Python pass afterward.

Payload filters (city/category/price) run inside the search itself, so
`limit=20` means "the best 20 that match", not "the best 20, some of which
match": ask for cheap food in Lisbon and the dense+BM25 pass returns its best
candidates that are actually cheap food in Lisbon, not the best candidates
from the whole corpus with a filter applied afterward.

BM25 lives here too, as a native sparse vector on the same point as the dense
one, with IDF computed by Qdrant from live collection statistics
(Modifier.IDF).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient, models

from config import (
    BM25_AVG_LEN,
    BM25_MODEL,
    BUILD_CMD,
    DENSE_VECTOR,
    EMBED_DIM,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_TIMEOUT,
    QDRANT_URL,
    SPARSE_VECTOR,
    UPSERT_BATCH,
)

# Fixed namespace so point ids are a pure function of the record id — the same
# chunk always maps to the same point, making build() idempotent and a
# partial/interrupted upsert safe to re-run.
_ID_NAMESPACE = uuid.UUID("2f9a6c1e-3b4d-5e6f-8a90-b1c2d3e4f506")

# text/url/id are never filtered on; city_key is filtered on but never shown,
# so it's deliberately excluded from what gets returned to callers.
PAYLOAD_FIELDS = ["id", "text", "url", "city", "category", "price_level"]


class StoreUnavailable(FileNotFoundError):
    """Qdrant is not reachable at QDRANT_URL.  Fix: docker compose up -d

    Subclasses FileNotFoundError so app.py's existing `except FileNotFoundError`
    handler keeps working as a safety net even before it's updated to catch
    this specifically.
    """


class CollectionMissing(FileNotFoundError):
    """Qdrant is up but the collection is absent or empty.  Fix: build_index.py"""


def point_id(record_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, record_id))


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)


@lru_cache(maxsize=1)
def _bm25():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(
        model_name=BM25_MODEL,
        avg_len=BM25_AVG_LEN,  # measured from this corpus; fastembed's own default (256) is ~5x too high
        language="english",
    )


def _to_sparse(embedding) -> models.SparseVector:
    # embed() yields TF-saturated float weights; query_embed() yields deduped
    # token ids with int32 values — cast explicitly so both paths agree.
    return models.SparseVector(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )


def sparse_passages(texts: list[str]) -> list[models.SparseVector]:
    return [_to_sparse(e) for e in _bm25().embed(texts, batch_size=UPSERT_BATCH)]


def sparse_query(text: str) -> models.SparseVector:
    return _to_sparse(next(iter(_bm25().query_embed(text))))


def ensure_collection(*, recreate: bool = False) -> None:
    client = _client()
    if recreate and client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(collection_name=QDRANT_COLLECTION, timeout=60)

    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=EMBED_DIM,
                    # embedder already L2-normalises, so cosine == inner product
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    # fastembed emits only the TF half of BM25 on purpose; without
                    # this modifier the sparse branch silently degenerates to raw
                    # term frequency instead of full BM25 with IDF.
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        # No hnsw_config/indexing_threshold override. Measured: at this corpus
        # size Qdrant crosses the default indexing_threshold during a batched
        # upsert and builds a full HNSW graph (indexed_vectors_count ends up
        # equal to points_count), so search is approximate, not brute-force
        # exact. Left as the default rather than forced to
        # indexing_threshold=0: at ~3k points HNSW's default recall is
        # effectively indistinguishable from exact, and disabling it would
        # trade that for slower search as the corpus grows.

    # create_payload_index is a no-op when the schema already matches, so this
    # is safe to call on every ensure_collection(), recreate or not.
    for field in ("city_key", "category", "price_level"):
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )


@dataclass
class Store:
    client: QdrantClient
    collection: str

    @property
    def size(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count


def build(records: list[dict], vectors: np.ndarray) -> None:
    if len(records) != vectors.shape[0]:
        raise ValueError(f"{len(records)} records vs {vectors.shape[0]} vectors")
    if vectors.shape[1] != EMBED_DIM:
        raise ValueError(
            f"embedder produced {vectors.shape[1]}-dim vectors but EMBED_DIM={EMBED_DIM}; "
            f"set EMBED_DIM to match EMBED_MODEL"
        )

    client = _client()
    ensure_collection(recreate=True)  # a rebuild replaces the corpus wholesale;
    # leaving stale points around would poison the collection-global IDF stats.

    print(f"  encoding BM25 sparse vectors for {len(records)} chunks...")
    sparse = sparse_passages([r["text"] for r in records])

    for start in range(0, len(records), UPSERT_BATCH):
        batch = records[start : start + UPSERT_BATCH]
        points = [
            models.PointStruct(
                id=point_id(rec["id"]),
                vector={
                    DENSE_VECTOR: vectors[start + j].tolist(),  # must be a plain list, not ndarray
                    SPARSE_VECTOR: sparse[start + j],
                },
                payload={**rec, "city_key": rec["city"].lower()},
            )
            for j, rec in enumerate(batch)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        print(f"  upserted {min(start + UPSERT_BATCH, len(records))}/{len(records)}")


def load() -> Store:
    client = _client()
    try:
        exists = client.collection_exists(QDRANT_COLLECTION)
    except Exception as exc:
        raise StoreUnavailable(
            f"Cannot reach Qdrant at {QDRANT_URL} ({type(exc).__name__}). "
            f"Start it with `docker compose up -d`."
        ) from exc

    if not exists:
        raise CollectionMissing(
            f"Collection '{QDRANT_COLLECTION}' does not exist at {QDRANT_URL}. "
            f"Run `{BUILD_CMD}` first."
        )

    count = client.count(collection_name=QDRANT_COLLECTION, exact=True).count
    if count == 0:
        raise CollectionMissing(
            f"Collection '{QDRANT_COLLECTION}' is empty. "
            f"Run `{BUILD_CMD}` first."
        )

    dim = client.get_collection(QDRANT_COLLECTION).config.params.vectors[DENSE_VECTOR].size
    if dim != EMBED_DIM:
        raise CollectionMissing(
            f"Collection '{QDRANT_COLLECTION}' stores {dim}-dim vectors but EMBED_DIM={EMBED_DIM}. "
            f"EMBED_MODEL changed — re-run `{BUILD_CMD}`."
        )

    return Store(client=client, collection=QDRANT_COLLECTION)
