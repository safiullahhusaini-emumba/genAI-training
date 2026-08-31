"""Single loaded instance of the embedding model.

BGE is asymmetric: queries get an instruction prefix, passages do not. Getting
this backwards is a silent ~5-point recall hit, so the two paths are separate
functions on purpose.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import EMBED_MODEL, QUERY_PREFIX


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed_passages(texts: list[str], batch_size: int = 64) -> np.ndarray:
    vecs = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # so inner product == cosine
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def embed_query(text: str) -> np.ndarray:
    vec = _model().encode(
        [QUERY_PREFIX + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype("float32")
