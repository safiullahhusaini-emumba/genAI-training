"""Central knobs. Everything tunable lives here so the pipeline files stay boring."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

for _d in (DATA_DIR, RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Groq --------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq deprecates model IDs fairly aggressively. If a call 404s, check
# https://console.groq.com/docs/models and update these two lines only.
MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b")  # prefs, judge, rerank
MODEL_SMART = os.getenv("GROQ_MODEL_SMART", "openai/gpt-oss-120b")  # final answer

# --- embeddings --------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Must match EMBED_MODEL. Qdrant rejects dimension mismatches at upsert time,
# and store.load() re-checks it against the live collection on every startup.
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
# BGE models were trained with an asymmetric prefix on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Qdrant --------------------------------------------------------------------
# A server now, not a library: `docker compose up -d` before anything else.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# "" would be sent as a literal empty api-key header. Coerce blank to None so
# "unset" means unset — works locally either way, breaks against Qdrant Cloud.
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "travel_chunks")
QDRANT_TIMEOUT = 30          # seconds; client default is 5s, cold-start upserts exceed it
UPSERT_BATCH = 256           # points per upsert call; payloads carry full chunk text

# Set to "1" by docker-compose.yml. There's no reliable way to detect this from
# inside the process — /.dockerenv is a Docker implementation detail, absent
# under Podman — so compose says so explicitly.
IN_CONTAINER = os.getenv("TRAVEL_RAG_IN_CONTAINER") == "1"

# Display only, never passed to a client. Inside a container QDRANT_URL is
# http://qdrant:6333, which resolves on the compose network and nowhere else;
# the sidebar renders a clickable /dashboard link and the browser is on the host.
QDRANT_DASHBOARD_URL = os.getenv("QDRANT_DASHBOARD_URL", QDRANT_URL)

# The fix command the UI and store.py print when the collection is missing. It
# differs between the host venv and the container, and getting it wrong sends a
# reviewer down a dead end.
BUILD_CMD = (
    "docker compose run --rm builder --from-corpus"
    if IN_CONTAINER
    else "python scripts/build_index.py"
)

DENSE_VECTOR = "dense"       # bge-small-en-v1.5, 384-d, cosine
SPARSE_VECTOR = "bm25"       # term-frequency sparse; IDF applied server-side

# --- lexical -------------------------------------------------------------------
BM25_MODEL = os.getenv("BM25_MODEL", "Qdrant/bm25")
# fastembed takes average document length as a fixed hyperparameter, where
# BM25Okapi derived it from the corpus. Measured over data/corpus.jsonl: ~49
# tokens after stemming and stopword removal. fastembed's own default of 256.0
# is ~5x too high here and under-penalises long chunks. Re-measure if
# CHUNK_MIN_CHARS / CHUNK_MAX_CHARS change.
BM25_AVG_LEN = float(os.getenv("BM25_AVG_LEN", "49"))

# --- chunking ----------------------------------------------------------------
CHUNK_MIN_CHARS = 220     # merge anything shorter into the next paragraph
CHUNK_MAX_CHARS = 1100    # hard split above this

# --- tagging -------------------------------------------------------------
# Category/price tags are LLM-classified in batches and cached by content hash,
# so re-running ingest after a chunking tweak only pays for genuinely new chunks.
TAG_CACHE_PATH = DATA_DIR / "tag_cache.json"
TAG_BATCH_SIZE = 25       # chunks per LLM tagging call

# --- retrieval ---------------------------------------------------------------
DENSE_K = 20              # candidates from the dense prefetch
BM25_K = 20               # candidates from the sparse (bm25) prefetch
RRF_K = 60                # reciprocal-rank-fusion constant
RERANK_INPUT = 12         # candidates handed to the LLM reranker
FINAL_K = 5               # chunks that reach the answer prompt

USE_HYDE = os.getenv("USE_HYDE", "true").lower() == "true"
JUDGE_PASS_THRESHOLD = 3  # need >= this many chunks scored "relevant" to proceed
