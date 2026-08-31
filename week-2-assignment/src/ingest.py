"""Fetch -> clean -> chunk -> tag.

Chunking strategy: paragraph-based. Paragraphs are the natural unit in travel
writing (one venue, one neighbourhood, one tip per block). Short paragraphs get
glued to the next one so we don't index a bare heading; anything over
CHUNK_MAX_CHARS is split on sentence boundaries.

Per-chunk category/price tags are LLM-classified (see `_tag_batch_llm`), sent
in batches of TAG_BATCH_SIZE rather than one call per chunk, and cached to
disk by content hash (TAG_CACHE_PATH) so re-running ingest after a chunking
tweak only pays for chunks that are actually new. Ten pages produce ~600
chunks; batched at 25/call that's ~24 calls on a cold cache and near-zero on
a warm one. The keyword-rule tagger below is kept only as the fallback for
whatever a batch's LLM call fails to classify (rate limit, bad JSON, etc.).
"""

from __future__ import annotations

import hashlib
import json
import re

import requests
import trafilatura

from config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    MODEL_FAST,
    RAW_DIR,
    TAG_BATCH_SIZE,
    TAG_CACHE_PATH,
)
from src.llm import chat_json
from src.sources import CATEGORIES, PRICE_LEVELS, SOURCES

HEADERS = {"User-Agent": "preference-aware-travel-rag/0.1 (assignment project)"}

CATEGORY_RULES = {
    "food": [
        "restaurant", "cafe", "café", "eat", "food", "dining", "kebab", "bakery",
        "breakfast", "lunch", "dinner", "beer", "bar", "pub", "market", "snack",
        "cuisine", "brunch", "coffee", "street food", "menu", "vegan",
    ],
    "art": [
        "museum", "gallery", "art", "exhibition", "mural", "street art",
        "sculpture", "painting", "collection", "artist", "contemporary",
        "installation", "biennale", "studio",
    ],
    "sightseeing": [
        "monument", "cathedral", "church", "palace", "tower", "square", "park",
        "bridge", "castle", "memorial", "landmark", "viewpoint", "walk", "gate",
        "ruins", "quarter", "district", "boat", "tour",
    ],
}

CHEAP_SIGNALS = [
    "cheap", "budget", "free", "affordable", "inexpensive", "low-cost",
    "no charge", "free entry", "free admission", "hostel", "street food",
    "under €", "under $", "bargain", "discount", "student ticket",
]
EXPENSIVE_SIGNALS = [
    "expensive", "michelin", "upscale", "luxury", "fine dining", "high-end",
    "pricey", "splurge", "gourmet", "five-star", "reservations required",
]


def _keyword_patterns(keywords: list[str]) -> list[re.Pattern]:
    """Word-boundary patterns, not substrings: plain `in`/`.count()` matching
    let "art" hit inside "start", "bar" hit inside "barrier", etc. The trailing
    `s?` keeps simple plurals ("eat" -> "eats", "restaurant" -> "restaurants")
    matching, since that coverage came for free with substring matching."""
    return [re.compile(r"(?<!\w)" + re.escape(kw) + r"s?(?!\w)") for kw in keywords]


_CATEGORY_PATTERNS = {cat: _keyword_patterns(kws) for cat, kws in CATEGORY_RULES.items()}
_CHEAP_PATTERNS = _keyword_patterns(CHEAP_SIGNALS)
_EXPENSIVE_PATTERNS = _keyword_patterns(EXPENSIVE_SIGNALS)

TAG_SYSTEM = """You classify short passages from travel guides.

For each numbered passage, assign:
- "category": one of "food", "art", "sightseeing" -- food = restaurants, cafes,
  bars, markets, dishes; art = museums, galleries, street art, exhibitions;
  sightseeing = landmarks, parks, walking, history, transport, or anything else.
- "price_level": one of "cheap", "medium", "expensive" -- the cost implied by
  the passage. If price isn't mentioned or is unclear, use "medium".

Return exactly this shape, with exactly one entry per passage index given:
{"tags": [{"i": <passage index>, "category": "...", "price_level": "..."}, ...]}"""

_WS = re.compile(r"[ \t\u00a0]+")
_MULTI_NL = re.compile(r"\n{2,}")
_SENT = re.compile(r"(?<=[.!?])\s+")


def fetch(url: str) -> str | None:
    """Download a page and strip the boilerplate. Cached to data/raw."""
    cache = RAW_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! fetch failed for {url}: {exc}")
        return None

    text = trafilatura.extract(
        resp.text,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not text:
        print(f"  ! no extractable text at {url}")
        return None

    text = clean(text)
    cache.write_text(text, encoding="utf-8")
    return text


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = _WS.sub(" ", text)
    # drop wiki edit markers and citation brackets
    text = re.sub(r"\[\s*edit\s*\]", "", text, flags=re.I)
    text = re.sub(r"\[\d{1,3}\]", "", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if len(ln) > 1]
    return _MULTI_NL.sub("\n\n", "\n".join(lines)).strip()


def paragraphs(text: str) -> list[str]:
    """Paragraph chunker with a min-merge and a max-split."""
    raw = [p.strip() for p in text.split("\n") if p.strip()]
    merged: list[str] = []
    buf = ""

    for para in raw:
        buf = f"{buf}\n{para}".strip() if buf else para
        if len(buf) >= CHUNK_MIN_CHARS:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = f"{merged[-1]}\n{buf}"
        else:
            merged.append(buf)

    out: list[str] = []
    for chunk in merged:
        if len(chunk) <= CHUNK_MAX_CHARS:
            out.append(chunk)
            continue
        piece = ""
        for sent in _SENT.split(chunk):
            if len(piece) + len(sent) + 1 > CHUNK_MAX_CHARS and piece:
                out.append(piece.strip())
                piece = sent
            else:
                piece = f"{piece} {sent}".strip()
        if piece:
            out.append(piece.strip())
    return out


def tag_category(text: str, fallback: str) -> str:
    """Deterministic fallback for whatever the LLM batch doesn't classify."""
    low = text.lower()
    scores = {
        cat: sum(len(p.findall(low)) for p in pats) for cat, pats in _CATEGORY_PATTERNS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else fallback


def tag_price(text: str, fallback: str) -> str:
    """Deterministic fallback for whatever the LLM batch doesn't classify."""
    low = text.lower()
    cheap = sum(len(p.findall(low)) for p in _CHEAP_PATTERNS)
    pricey = sum(len(p.findall(low)) for p in _EXPENSIVE_PATTERNS)
    if cheap > pricey and cheap > 0:
        return "cheap"
    if pricey > cheap and pricey > 0:
        return "expensive"
    return fallback


def _chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_tag_cache() -> dict:
    if TAG_CACHE_PATH.exists():
        try:
            return json.loads(TAG_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_tag_cache(cache: dict) -> None:
    TAG_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _tag_batch_llm(chunks: list[str]) -> list[dict | None]:
    """One LLM call classifies a whole batch. Returns None per-index for
    anything the model omits or gets wrong, so the caller can fall back."""
    numbered = "\n\n".join(f"[{i}] {c[:400]}" for i, c in enumerate(chunks))
    try:
        raw = chat_json(
            TAG_SYSTEM,
            numbered,
            model=MODEL_FAST,
            temperature=0.0,
            max_tokens=min(4096, 60 * len(chunks) + 200),
        )
    except Exception as exc:  # network, auth, rate limit
        print(f"  ! LLM tagging batch failed ({type(exc).__name__}); using keyword rules for this batch")
        return [None] * len(chunks)

    results: list[dict | None] = [None] * len(chunks)
    for item in (raw.get("tags") or []):
        i = item.get("i")
        cat = item.get("category")
        price = item.get("price_level")
        if isinstance(i, int) and 0 <= i < len(chunks) and cat in CATEGORIES and price in PRICE_LEVELS:
            results[i] = {"category": cat, "price_level": price}
    return results


def _tag_all(texts: list[str], fallback_categories: list[str], fallback_prices: list[str]) -> list[tuple[str, str]]:
    """Tag every chunk, reusing the on-disk cache and only sending cache
    misses to the LLM, in batches of TAG_BATCH_SIZE."""
    cache = _load_tag_cache()
    hashes = [_chunk_hash(t) for t in texts]
    to_query = [i for i, h in enumerate(hashes) if h not in cache]

    if to_query:
        print(f"  tagging {len(to_query)} new chunks via LLM ({len(texts) - len(to_query)} already cached)")
        for start in range(0, len(to_query), TAG_BATCH_SIZE):
            batch_idx = to_query[start : start + TAG_BATCH_SIZE]
            results = _tag_batch_llm([texts[i] for i in batch_idx])
            for i, res in zip(batch_idx, results):
                if res is not None:
                    cache[hashes[i]] = res
            _save_tag_cache(cache)  # persist after every batch so an interrupted run loses no progress
    else:
        print("  all chunks already tagged (cache hit)")

    out: list[tuple[str, str]] = []
    for i, h in enumerate(hashes):
        cached = cache.get(h)
        if cached:
            out.append((cached["category"], cached["price_level"]))
        else:
            out.append((
                tag_category(texts[i], fallback_categories[i]),
                tag_price(texts[i], fallback_prices[i]),
            ))
    return out


def build_corpus() -> list[dict]:
    """Returns a flat list of chunk records ready for embedding."""
    entries: list[dict] = []
    fallback_categories: list[str] = []
    fallback_prices: list[str] = []
    for src in SOURCES:
        print(f"- {src['url']}")
        text = fetch(src["url"])
        if not text:
            continue
        chunks = paragraphs(text)
        print(f"  {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            entries.append(
                {
                    "id": f"{hashlib.sha1(src['url'].encode()).hexdigest()[:10]}-{i:04d}",
                    "text": chunk,
                    "url": src["url"],
                    "city": src["city"],
                }
            )
            fallback_categories.append(src["category"])
            fallback_prices.append(src["price_level"])

    print(f"\nTagging {len(entries)} chunks...")
    tags = _tag_all([e["text"] for e in entries], fallback_categories, fallback_prices)
    for entry, (category, price) in zip(entries, tags):
        entry["category"] = category
        entry["price_level"] = price
    return entries


def save_corpus(records: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
