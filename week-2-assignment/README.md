# Preference-Aware Travel RAG Assistant

Answers travel questions like *"3-day Berlin trip with cheap food and art"* from a small
indexed corpus of travel pages — and shows its working at every step.

```
query → preferences → filtered hybrid retrieval → rerank → context judge → grounded answer
```

Groq for every LLM call. BGE-small for embeddings. Qdrant for retrieval — dense and sparse
vectors in one collection, filtered and fused server-side. Streamlit for the UI.

![architecture](docs/architecture.svg)

---

## Quick start

Everything runs in Docker

```bash
git clone <your-repo-url> && cd travel-rag

cp .env.example .env            # paste your key from https://console.groq.com/keys
# leave QDRANT_API_KEY blank — local Qdrant runs without one

docker compose up -d --build    # Qdrant + the UI; first build is a few minutes (torch)
docker compose ps               # wait for both qdrant and app to say (healthy)

docker compose run --rm builder --from-corpus   # ~2 min
```

Open http://localhost:8501.

The collection lives in a Docker named volume, not under `data/` — `docker volume ls` will
show it, alongside `travel-rag_model_cache`, which holds the downloaded embedding models
so they aren't re-fetched on every container start. `docker compose down` stops everything
and keeps both volumes; `docker compose down -v` deletes them — the next `builder` run
rebuilds the collection from `data/corpus.jsonl` in about two minutes and re-downloads
~130 MB of model weights. Nothing irreplaceable is in there.

**Editing code:** `docker compose up` bind-mounts the whole repo root read-only over
the built image (via `docker-compose.override.yml`, which explains why it's the whole
tree and not a file at a time), so changes are picked up live — Streamlit reruns on
save. `./data` stays read-write, nested inside that mount.
`docker compose -f docker-compose.yml up` (explicitly skipping the override) runs the
baked image standalone, the way it would ship.

---

## Design decisions

### Chunking — paragraph-based, with a min-merge and a max-split

Implemented in `src/ingest.py:paragraphs()`, driven by two knobs in `config.py`:
`CHUNK_MIN_CHARS = 220` and `CHUNK_MAX_CHARS = 1100`.

The unit of chunking is the **paragraph**, not a fixed-size sliding window. Travel
writing is already structured that way — one venue, one neighbourhood, or one tip per
paragraph — so splitting on paragraph boundaries keeps each chunk topically self-contained
without any semantic-similarity chunking machinery. Fixed-size windows would routinely cut
a venue description in half; paragraph boundaries in this kind of source text essentially
never do.

Two passes correct the raw paragraph split:

1. **Min-merge (`CHUNK_MIN_CHARS`, 220 chars).** Raw paragraphs are walked in order and
   concatenated into a running buffer until the buffer reaches 220 characters, then the
   buffer is flushed as one chunk. This exists because travel pages are full of short
   paragraphs that are useless alone — a heading, a one-line intro, a bare address — and
   indexing those as standalone chunks would waste a vector slot on something with nothing
   for the embedding model to grab onto. Merging glues each of those onto the paragraph(s)
   that follow it instead of discarding or separately indexing them. Any leftover buffer at
   the end of a page gets appended to the last completed chunk rather than shipped as an
   under-sized fragment on its own.
2. **Max-split (`CHUNK_MAX_CHARS`, 1100 chars).** Anything still over 1100 characters after
   merging — a long, detailed passage about one place — is split on sentence boundaries
   (`re.split(r"(?<=[.!?])\s+", ...)`), packing whole sentences into a piece until the next
   one would overflow the limit. This caps chunk size for retrieval and reranking without
   ever cutting a sentence in half, which a naive character-count split would do.

**No overlap between chunks.** Standard sliding-window RAG chunking usually overlaps
adjacent chunks (e.g. 10–20%) to avoid losing context at a cut point. That trade-off is
there to compensate for cutting at an arbitrary character offset. Here the cut points are
already paragraph and sentence boundaries chosen because they're natural breaks in the
content, so there's very little context to lose at the seam — and skipping overlap avoids
inflating the chunk count (and therefore the per-chunk LLM tagging cost and the embedding
index size) for redundant duplicate text.

**Tagging happens after chunking, not as part of it.** Each finished chunk is
LLM-classified into a `category` (food / art / sightseeing) and a `price_level` (cheap /
medium / expensive) — batched `TAG_BATCH_SIZE` chunks per call rather than one call per
chunk, and cached to disk by content hash (`sha1` of the chunk text) so re-running ingest
after tweaking `CHUNK_MIN_CHARS`/`CHUNK_MAX_CHARS` only pays the LLM for chunks that
actually changed. A deterministic keyword-rule tagger (`tag_category` / `tag_price` in
`src/ingest.py`) is the fallback for whatever a batch's LLM call fails to classify (rate
limit, malformed JSON, model omits an index).

The tradeoff: this strategy is tuned to source pages that are already reasonably
well-structured prose (Wikipedia/guide-style pages with real paragraph breaks). It would
do much less well on unstructured input — a wall of un-paragraphed text, or a page that's
mostly tables/lists — since the whole approach leans on paragraph breaks being meaningful.

### Preference extraction — LLM with a rule-based fallback

`src/preferences.py` turns the query into `{city, categories, price_level, duration_days,
keywords}`. One LLM call does the extraction; a regex/keyword-based `_rules()` fallback
covers Groq being down, rate-limited, or returning unparseable JSON — and also patches the
LLM path itself, since the fast model drops the city about 1 in 15 times.

City names get fuzzy-matched (`difflib`, cutoff `0.85`) against the known city list so a
typo ("pehawar") still resolves to the indexed city ("Peshawar") instead of silently
matching zero points — the city filter is never relaxed downstream, so an unmatched city
means a hard refusal. The `0.85` cutoff was calibrated against this corpus: real typos
score 0.87–0.94, the closest an absent city gets is 0.55.

### Qdrant collection

One collection, `travel_chunks`, holding both the dense vector and a native BM25 sparse
vector per point, with payload indexes on `city_key`/`category`/`price_level` so filters
run inside the search rather than after it. Point IDs are a UUID5 hash of the chunk's own
id, so upserts are idempotent and a rebuild is safe to re-run. `build_index.py` always
rebuilds the whole collection rather than upserting incrementally — leftover points from a
previous corpus would skew BM25's corpus-wide IDF stats. Vector search uses Qdrant's HNSW
index.

### Embedding model — `BAAI/bge-small-en-v1.5`

- **384 dimensions, ~33M params, ~130 MB.** Runs on CPU in a fraction of a second per
  batch, which matters when the whole thing has to survive a laptop demo without a GPU.
- **Punches above its size on retrieval specifically.** It sits near the top of the MTEB
  retrieval band for its parameter count — better than `all-MiniLM-L6-v2` at the same
  latency, and it isn't a general-purpose sentence-similarity model repurposed for search.
- **Asymmetric by design.** BGE was trained with an instruction prefix on the query side
  only, which fits the RAG shape exactly: short question, long passage. `embed_query()`
  and `embed_passages()` are separate functions in `src/embedder.py` for this reason —
  applying the prefix to passages, or forgetting it on queries, is a silent recall hit
  that no test catches.
- **Apache 2.0, fully local.** No second API dependency, no per-query embedding cost.

The obvious upgrade is `bge-base` or `e5-large`. For ~3,000 chunks the ceiling here is
chunking quality and reranking, not embedding capacity, so the extra 4× latency buys very
little.

### Retrieval — hybrid dense + sparse, fused with RRF

Dense retrieval alone loses on travel queries, because travel queries are full of proper
nouns. "Pergamonmuseum", "Markthalle Neun", "Tempelhofer Feld" — a 384-dim embedding
smears all three into a generic "place in Berlin" region of the space. The sparse side
matches them exactly. Meanwhile sparse search alone is useless on "somewhere atmospheric
for a rainy afternoon", which has no keyword overlap with anything.

The two are combined with **reciprocal rank fusion** rather than score averaging, because
dense search returns cosine in [0, 1] and the sparse side returns unbounded IDF-weighted
term scores — those aren't on the same scale, and normalising them per query is a fudge
that quietly changes ranking depending on how good the best hit happened to be. RRF only
looks at rank position, so the mismatch never arises.

Fusion now happens **inside Qdrant**: two prefetches — one against the dense vector, one
against the sparse — each carrying the same payload filter, fused by the server with
`k=60`. That `k` is the same constant that used to live in `retrieve.py`, passed through
rather than replaced, so the arithmetic is unchanged even though the location isn't. The
filter goes on each prefetch and not on the outer query, which is the whole point: a
filter on the outer query would run after fusion and reinvent the post-filter this
migration existed to delete.

The BM25 side's `avg_len` is overridden to `49` (measured tokens/chunk on this corpus)
instead of fastembed's default of `256`, which would under-penalise long chunks. It's
tied to the chunking config — re-measure if `CHUNK_MIN_CHARS`/`CHUNK_MAX_CHARS` change.

**HyDE** is available behind a UI toggle: one fast call writes a hypothetical guide
paragraph, and the dense side searches with `query + hypothetical` instead of the bare
query. It measurably helps vague queries and does roughly nothing for specific ones, so
it's optional rather than baked in. The sparse side always uses the real query —
searching lexically over invented text is a good way to retrieve invented places.

### Reranking

One batched LLM call scores all candidates 0–3, rather than N calls scoring one each.
Cheaper, faster, and the model ranks better when it can compare candidates side by side.
Ties break on fusion score, so retrieval rank still carries information inside a score band.

### Context judge

The reranker has already produced per-chunk relevance scores, so the gate is mostly just
counting them: ≥3 chunks scoring 2+ means proceed, 0 means stop. The LLM judge is only
invoked in the ambiguous middle (1–2 strong chunks). That keeps the check at roughly zero
extra calls on good queries.

On `context_insufficient`, the pipeline relaxes **category and price** filters, forces
HyDE on, and retries exactly once. **The city filter is never relaxed** — confidently
recommending a Prague restaurant to someone asking about Berlin is worse than admitting
the corpus doesn't cover it. If the retry also fails, it refuses, says what's missing, and suggests a question it
*can* answer — the refusal prompt is handed the indexed city list, so it can't
cheerfully propose another question about a city that isn't in the corpus.

### Model split — fast model for decisions, smart model for the answer

`gpt-oss-20b` (`MODEL_FAST`) runs preferences, reranking, judging and HyDE — every
intermediate decision. `gpt-oss-120b` (`MODEL_SMART`) is reserved for the one call the
user actually reads. Cheaper and faster where quality barely matters, stronger where it does.

---

## Repo layout

```
config.py                 all tunable knobs — models, k values, thresholds, Qdrant
Dockerfile                one image, shared by the app service and both jobs
.dockerignore
docker-compose.yml        qdrant + app service + two profile-gated jobs (builder, ask)
docker-compose.override.yml   dev-only whole-tree bind-mount for live reload (auto-loaded)
scripts/build_index.py    one-shot: fetch → chunk → embed → upsert
scripts/ask.py            one-shot: query the pipeline from the CLI
src/
  sources.py              the seed URLs + their source-level metadata
  ingest.py               fetch, clean, paragraph-chunk, LLM-tag (cached; keyword fallback)
  embedder.py             BGE wrapper, query/passage paths kept separate
  store.py                Qdrant client, collection schema, payload filters
  llm.py                  the only place a Groq call is made
  preferences.py          query → preferences JSON (LLM + rule fallback)
  retrieve.py             HyDE, dense+sparse prefetch, server-side RRF and filtering
  rerank.py               batched LLM reranker
  judge.py                context sufficiency gate
  generate.py             grounded answer / grounded refusal
  pipeline.py             orchestration + the trace the UI renders
app.py                    Streamlit UI
docs/architecture.svg     the diagram above
```
