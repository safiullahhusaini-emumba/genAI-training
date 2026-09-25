# Agentic LinkedIn Content Curator

Turns a topic into a LinkedIn-ready post plus a matching image, through a **planner-driven
agentic system**. A Planner Agent reads the topic and decides,
per request, which tools to call and in what order; an async DAG executor runs that plan;
a Generator Agent writes the post; an Editor Agent critiques and rewrites it.

```
topic → Planner (LLM, emits a DAG) → DAG Executor (per-step-ready async scheduling)
                                            │
                        ┌───────────────────┼────────────────────┐
                        ▼                   ▼                    ▼
                   web_search          social_search        image_generator
                   (Tavily)            (DDGS)               (art-director LLM →
                        │                   │                 Cloudflare FLUX →
                        └─────────┬─────────┘                 Pollinations → PIL card)
                                  ▼
                             summarizer (LLM)
                                  ▼
                          content_generator (LLM)  ── Generator Agent
                                  ▼
                           content_editor (LLM)     ── Editor Agent
                                  ▼
                          final post + image → Streamlit
```


---

## Quick start (Docker Compose)

```bash
git clone <your-repo-url> && cd <repo>
cp .env.example .env       # fill in GROQ_API_KEY, TAVILY_API_KEY,
                            # CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN — see .env.example

docker compose up -d --build
docker compose ps          # wait for both api and ui to say (healthy)
```

Open http://localhost:8502. The API alone is at http://localhost:8000 (`/plan`, `/execute`,
`/execute/stream`, `/health`) — both endpoints work with no UI involved; see **API
reference** below for curl examples.

**Ports:** the API publishes **8000** and the UI **8502**, both bound to loopback.

---

## Agent roles

| Agent | File | Role |
|---|---|---|
| **Planner** (Core Brain) | `src/planner.py` | Reads the topic, decides which tools to call and in what order, emits a DAG (`{"steps": [{"step", "tool", "depends_on", "args"}]}`). Model: `gpt-oss-20b`. |
| **Executor(s) / tools** | `src/tools/*.py` | `web_search` (Tavily), `social_search` (DDGS), `summarizer` (LLM), `image_generator` (art-director LLM + 3-tier image fallback). Each has a Pydantic arg schema, returns a structured dict, and is independently callable (`python -m src.tools.web_search "<topic>"`). |
| **Generator** | `src/tools/generator.py` | Takes aggregated content, writes `{headline, body, hashtags}`. Refuses rather than fabricates if every dependency is degraded. Model: `gpt-oss-120b`. |
| **Editor** | `src/tools/editor.py` | Critiques the draft (`gpt-oss-20b`, a decision), then rewrites it (`gpt-oss-120b`, what the user reads) addressing the critique. Passes a refusal straight through unedited. |

The **DAG executor** (`src/executor.py`) and **validator** (`src/validator.py`) aren't
agents.

---

## Planner output schema + execution logic

Schema (`src/plan_schema.py`):

```json
{"steps": [
  {"step": 1, "tool": "web_search", "depends_on": [], "args": {"time_range": "month"}},
  {"step": 2, "tool": "social_search", "depends_on": [], "args": {}},
  {"step": 3, "tool": "summarizer", "depends_on": [1, 2], "args": {}},
  {"step": 4, "tool": "content_generator", "depends_on": [3], "args": {}},
  {"step": 5, "tool": "content_editor", "depends_on": [4], "args": {}},
  {"step": 6, "tool": "image_generator", "depends_on": [], "args": {}}
]}
```

**Execution logic (`src/executor.py`) — per-step readiness, not level-based waves.** One `asyncio.Task` is created per step, and each awaits
*only its own* dependency tasks:

```python
for step in topological_order:
    tasks[step.step] = asyncio.create_task(run_step(step))   # awaits tasks[d] for d in step.depends_on
```

A real run's trace (`start_ms`–`end_ms`, from the debug panel):

```
step 1 web_search        wave=0  ok     0-1104ms
step 2 social_search     wave=0  ok     0-4154ms
step 3 summarizer        wave=1  ok  4154-5199ms   # started the instant 1+2 finished...
step 4 content_generator wave=2  ok  5199-6456ms
step 5 content_editor    wave=3  ok  6457-8631ms
step 6 image_generator   wave=0  ok     0-6026ms   # ...while THIS was still running
outcome: answered | total_ms: 8632
```

---

## Why a planner is actually needed

This is the part a hard-coded pipeline can't do, demonstrated with three real planner
outputs from this system, unedited:

**"recent trends in GenAI agents for backend engineers"** — fast-moving topic, full 6-step
plan, `web_search` tuned to `time_range: "month"`.

**"fundamentals of event-driven architecture"** — evergreen topic, same 6-step shape, but
`web_search` gets `time_range: "year"` instead — a wider net, because narrowing to a month
would miss most of what actually explains the fundamentals.

**"why I left consulting for a startup"** — personal-opinion topic. The planner emitted a
**3-step plan**: `content_generator → content_editor`, plus `image_generator`, no search at all. Skipping search here
isn't a missing feature — searching for "why people leave consulting" and blending strangers'
opinions into a first-person post would make it *worse*. A fixed pipeline that always runs
"search → summarize → generate → edit" cannot make this call; a planner reading the topic can. Also, since image generation for a post is must, the validator pass makes sure that this type of query ALWAYS produces an image.

### The planner is not trusted blindly

An LLM-written DAG fails in specific, repeatable ways. `src/validator.py` is the
deterministic safety net between "the planner said so" and "the executor runs it" — every
repair is recorded and shows up in the trace / debug panel:

| Failure mode | Repair |
|---|---|
| Hallucinated tool name | Drop the step |
| Dependency on a step that doesn't exist | Drop the dependency |
| Duplicate step ids | Renumber (first occurrence wins) |
| A dependency cycle | Reject the whole plan, use the default plan |
| Missing `content_generator`/`content_editor` | Inject them at the end |
| Missing `image_generator` | Inject it with `depends_on: []` |
| Too many steps | Reject the whole plan, use the default plan |
| Empty/unparseable LLM response | Caught one layer up (`src/llm.py`), use the default plan |

---

## Failure cases observed

### The planner's "skip search" permission quietly generalized into "skip the image too"

For personal-opinion topics like *"why I left consulting for a startup"*, the planner's
system prompt permitted skipping `web_search`/`social_search`, and `gpt-oss-20b` produced
2-step plans (`content_generator → content_editor`) that dropped `image_generator` too — for a
system whose entire pitch is "post plus a matching image."

The model wasn't really disobeying; the prompt led it there. The only "must include" rule
named `content_generator`/`content_editor`. The image rule only said *how* to wire
`image_generator` (`depends_on: []`), never *that* it was required. The skip-search rule said
to wire `content_generator` "directly to nothing" without mentioning the image, and the only
worked example was the full 6-step research plan, so a no-search plan with an image had
never been shown.

Nothing downstream caught it either: `src/validator.py` re-injected a missing
`content_generator`/`content_editor` but had no equivalent for `image_generator`. So the UI
rendered a text-only post with no warning — the plan was internally consistent, just
incomplete.

**Fix, in two layers:**

1. **Prompt (root cause), `src/planner.py`:** a rule that every plan must include exactly one
   `image_generator` ("skipping research never means skipping the image"), plus a reminder
   inside the skip-search rule itself.
2. **Validator (backstop), `src/validator.py`:** `validate_and_repair` now injects
   `image_generator` (`depends_on: []`) if it's still missing, the same way it handles
   `content_generator`/`content_editor`, and records a repair note. The prompt makes the
   omission rare; the validator makes it impossible.

---

## Design decisions

- **Orchestration:** LangChain primitives (`ChatGroq`, `@tool`-shaped tool functions) plus a
  hand-written async DAG executor.
- **Search:** Tavily (`web_search`, general topic + advanced depth) and DDGS (`social_search`, site-restricted to
  x.com/reddit.com/news.ycombinator.com). A dedicated X/Twitter search would be the natural
  fit; the real X API v2 search endpoint is paid (~$200/mo) and not used here — DDGS is the honest,
  keyless substitute, restricted to platforms where people actually discuss things.
- **Image generation:** three-tier fallback — Cloudflare Workers AI (`flux-1-schnell`, 10k
  free neurons/day, no card, confirmed working live) → Pollinations (keyless, no signup) →
  a locally rendered typographic card (PIL, no network at all).
- **Structured output:** `ChatGroq.with_structured_output(method="json_schema")` — Groq's
  own server-side schema enforcement, verified working on both `gpt-oss-20b` and
  `gpt-oss-120b`.
- **Editor:** critique → revise, not a single "improve this" rewrite, so the UI can show
  *what* the editor found plus a before/after of what actually changed.
- **UI flow:** two calls (`/plan` then `/execute`), with the plan rendered as an editable
  checklist between them.
- **Refusal:** refuse only if every dependency is degraded, not on any single tool failure.

---

## API reference

```bash
curl -s localhost:8000/plan -H 'content-type: application/json' \
  -d '{"topic": "recent trends in GenAI agents for backend engineers"}' | python3 -m json.tool

curl -s localhost:8000/execute -H 'content-type: application/json' \
  -d '{"topic": "recent trends in GenAI agents for backend engineers"}' | python3 -m json.tool

# /execute accepts an optional "plan" (the exact shape /plan returns) to run a
# specific, possibly user-edited, DAG instead of planning internally:
curl -s localhost:8000/execute -H 'content-type: application/json' \
  -d '{"topic": "...", "plan": {"steps": [...]}}'
```

`/revise` re-runs **only** the Editor Agent over a post that already exists — the UI's
"Revise again" button. Re-running the whole DAG to improve wording would re-issue both
searches and regenerate the image (~10s plus image budget) for a change that only touches
text, so this path skips all of that:

```bash
curl -s localhost:8000/revise -H 'content-type: application/json' \
  -d '{"topic": "...", "post": {"headline": "...", "body": "...", "hashtags": ["..."]}}'
```

It reuses `src/tools/editor.py` unchanged by handing it a synthetic `content_generator`
dependency rather than special-casing the tool — which is what "each tool is independently
callable" is supposed to buy you.

> **Caveat worth knowing:** each press critiques the *current* text, so repeated presses
> refine rather than repeat — but they are grounded only in the previous version, not in the
> original search results. The rewrite prompt forbids introducing any statistic, number, or
> product name not already present, which is what keeps successive passes from drifting away
> from the sourced material.

Every tool is also directly runnable, satisfying "callable independently":

```bash
python -m src.tools.web_search "recent trends in GenAI agents for backend engineers"
python -m src.tools.social_search "GenAI agents for backend engineers"
python -m src.tools.image_gen "recent trends in GenAI agents for backend engineers"
python -m src.planner "why I left consulting for a startup"
```


---

## Repo layout

```
config.py                    every knob, each with a comment saying WHY
requirements.txt             pinned versions, each with a reason
Dockerfile                   one image, shared by the api and ui services
docker-compose.yml           api (:8000) + ui (:8502)
docker-compose.override.yml  dev-only bind mount for live reload (auto-loaded)
api/
  main.py                    /health /plan /execute /execute/stream /revise
  schemas.py                 request/response models
src/
  llm.py                     the ONLY place an LLM is called; chat_json() + schema enforcement
  plan_schema.py             Step/Plan models + the default plan
  planner.py                 Planner Agent
  validator.py               deterministic repair of a planner's DAG
  executor.py                per-step-readiness async DAG executor + trace
  types.py                   StepResult, shared between tools and the executor
  tools/
    web_search.py            Tavily
    social_search.py         DDGS
    summarizer.py            condenses raw search results into key points
    generator.py             Generator Agent
    editor.py                Editor Agent
    image_gen.py             art director + 3-tier image fallback
ui/app.py                    Streamlit UI
```

## Limitations

- Cloudflare's free tier is 10,000 neurons/day shared across every model call on the
  account — heavy use can exhaust it; Pollinations and the local card exist for exactly
  that reason.
- DDGS is an unofficial wrapper around DuckDuckGo's HTML; it can break when DuckDuckGo
  changes markup, independent of anything in this codebase.
- The planner's tool-argument tuning (e.g. `time_range`) is only as good as the one-shot LLM
  call that produces it — the validator repairs *structural* mistakes, not judgment calls
  that are merely suboptimal rather than invalid.
