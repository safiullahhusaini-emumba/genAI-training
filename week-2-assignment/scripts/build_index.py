"""Run once: fetch the seed URLs, chunk, embed, upsert into Qdrant.

    python scripts/build_index.py
    python scripts/build_index.py --from-corpus   # re-embed data/corpus.jsonl, no fetch/tagging

Raw page text is cached in data/raw, so re-runs after a chunking change are
fast and don't hammer Wikivoyage. `--from-corpus` skips fetch and tagging
entirely by re-reading data/corpus.jsonl (which already carries every chunk's
LLM-assigned category/price tag) — the flag to reach for after a schema or
filter change, or to repopulate Qdrant without a Groq key.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, QDRANT_COLLECTION, QDRANT_URL  # noqa: E402
from src import store  # noqa: E402
from src.embedder import embed_passages  # noqa: E402
from src.ingest import build_corpus, save_corpus  # noqa: E402


def _check_qdrant() -> None:
    """Fail fast: a ~90-second embed is wasted if the container isn't up."""
    try:
        store._client().get_collections()
    except Exception as exc:
        sys.exit(
            f"Cannot reach Qdrant at {QDRANT_URL} ({type(exc).__name__}).\n"
            f"Start it with `docker compose up -d` before running this script."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from-corpus",
        action="store_true",
        help="Re-embed data/corpus.jsonl instead of re-fetching and re-tagging. "
        "No network, no LLM calls — use this to (re)populate Qdrant.",
    )
    args = ap.parse_args()

    _check_qdrant()

    if args.from_corpus:
        path = DATA_DIR / "corpus.jsonl"
        if not path.exists():
            sys.exit(f"{path} not found; run without --from-corpus")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        print(f"Loaded {len(records)} chunks from {path} (no fetch, no tagging)")
    else:
        print("Fetching and chunking sources...")
        records = build_corpus()
        if not records:
            sys.exit("No records produced. Check your network connection.")
        save_corpus(records, DATA_DIR / "corpus.jsonl")

    print(f"\n{len(records)} chunks from {len({r['url'] for r in records})} pages")
    print("  by category:   ", dict(Counter(r["category"] for r in records)))
    print("  by price level:", dict(Counter(r["price_level"] for r in records)))
    print("  by city:       ", dict(Counter(r["city"] for r in records)))

    print("\nEmbedding (first run downloads ~130 MB of model weights)...")
    vectors = embed_passages([r["text"] for r in records])

    print(f"\nWriting to Qdrant collection '{QDRANT_COLLECTION}'...")
    store.build(records, vectors)
    print(f"Done. {len(records)} points in '{QDRANT_COLLECTION}' at {QDRANT_URL}")


if __name__ == "__main__":
    main()
