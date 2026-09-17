# Agentic LinkedIn Content Curator

Turns a topic into a LinkedIn-ready post plus a matching image, through a **planner-driven
agentic system** — not a hard-coded pipeline. A Planner Agent reads the topic and decides,
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

House rules the code holds to throughout: one commented `config.py` where every knob says
*why* it is that value, one module per stage, one choke point for every LLM call, a
deterministic fallback for every LLM decision, and a trace that is a first-class return
value rather than a log line. Every non-obvious decision below is recorded with its
alternatives and its tradeoff, not just its outcome.

---

## Quick start (Docker Compose)

```bash
git clone <your-repo-url> && cd <repo>
cp .env.example .env       # fill in GROQ_API_KEY, TAVILY_API_KEY,
                            # CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN — see .env.example
                            # for where to get each one free, no card

docker compose up -d --build
docker compose ps          # wait for both api and ui to say (healthy)
```

Open http://localhost:8502. The API alone is at http://localhost:8000 (`/plan`, `/execute`,
`/execute/stream`, `/health`) — both endpoints work with no UI involved; see **API
reference** below for curl examples.

`docker compose -f docker-compose.yml up` (skipping the auto-loaded dev override) runs the
image exactly as it would ship. There is no database or other backing service to stand up —
the two containers are the whole system.

**Ports:** the API publishes **8000** and the UI **8502**, both bound to loopback. 8502
rather than Streamlit's default 8501, because that default is very often already bound on a
developer machine and a port clash on first `up` is a poor first impression. Only the *host*
side differs — inside its container the UI still listens on 8501, so nothing internal
(healthchecks, `API_BASE_URL`) depends on the remapping.

---

## Agent roles

| Agent | File | Role |
|---|---|---|
| **Planner** (Core Brain) | `src/planner.py` | Reads the topic, decides which tools to call and in what order, emits a DAG (`{"steps": [{"step", "tool", "depends_on", "args"}]}`). Model: `gpt-oss-20b`. |
| **Executor(s) / tools** | `src/tools/*.py` | `web_search` (Tavily), `social_search` (DDGS), `summarizer` (LLM), `image_generator` (art-director LLM + 3-tier image fallback). Each has a Pydantic arg schema, returns a structured dict, and is independently callable (`python -m src.tools.web_search "<topic>"`). |
| **Generator** | `src/tools/generator.py` | Takes aggregated content, writes `{headline, body, hashtags}`. Refuses rather than fabricates if every dependency is degraded. Model: `gpt-oss-120b`. |
| **Editor** | `src/tools/editor.py` | Critiques the draft (`gpt-oss-20b`, a decision), then rewrites it (`gpt-oss-120b`, what the user reads) addressing the critique. Passes a refusal straight through unedited. |

The **DAG executor** (`src/executor.py`) and **validator** (`src/validator.py`) aren't
agents — they're the deterministic scaffolding the assignment's diagram doesn't name but
requires: something has to actually run the planner's DAG, and something has to catch the
planner when it's wrong.

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

**Execution logic (`src/executor.py`) — per-step readiness, not level-based waves.** The
obvious implementation groups steps into topological levels and `asyncio.gather`s each level
before starting the next. That's wrong here: step 3 (`summarizer`) depends on `[1, 2]`, not
on step 6 (`image_generator`, `depends_on: []`) — but a level scheduler would put all three
in "level 0" and make step 3 wait for whichever of the three is slowest, purely because they
happen to share a level. Instead, one `asyncio.Task` is created per step, and each awaits
*only its own* dependency tasks:

```python
for step in topological_order:
    tasks[step.step] = asyncio.create_task(run_step(step))   # awaits tasks[d] for d in step.depends_on
```

This is measured, not asserted — a real run's trace (`start_ms`–`end_ms`, from the debug panel):

```
step 1 web_search        wave=0  ok     0-1104ms
step 2 social_search     wave=0  ok     0-4154ms
step 3 summarizer        wave=1  ok  4154-5199ms   # started the instant 1+2 finished...
step 4 content_generator wave=2  ok  5199-6456ms
step 5 content_editor    wave=3  ok  6457-8631ms
step 6 image_generator   wave=0  ok     0-6026ms   # ...while THIS was still running
outcome: answered | total_ms: 8632
```

Read step 3 against step 6. `summarizer` started at **4154ms**, while `image_generator` was
still running and didn't finish until **6026ms**. A level-based scheduler would have put
steps 1, 2 and 6 in the same level and made `summarizer` wait for the slowest of them — so
it couldn't have started until 6026ms, **1872ms later**, pushing the run from 8632ms to
10502ms. Same work, same tools, ~1.9s slower, purely from a scheduling choice.

That is the entire argument for per-step readiness, and it's checkable in any run's debug
panel: whenever a no-dependency step (here, image generation) is slower than the search
steps, a level scheduler stalls the chain behind work the chain has nothing to do with.
`wave` is still recorded per step, but purely as a topological-level label for the UI's Gantt
grouping — it plays no role in scheduling.

A tool is never allowed to kill the whole run: every exception is caught in the executor and
turned into a failed `StepResult`, so one dead API can't take down steps that don't depend
on it.

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
**2-step plan**: `content_generator → content_editor`, no search at all. Skipping search here
isn't a missing feature — searching for "why people leave consulting" and blending strangers'
opinions into a first-person post would make it *worse*. A fixed pipeline that always runs
"search → summarize → generate → edit" cannot make this call; a planner reading the topic can.

The tool **set** and the tool **arguments** both change with the topic, per request. That's
the concrete claim "planner-driven, not a pipeline" is making, and it's checkable: run
`python -m src.planner "<any topic>"` and read the DAG it returns.

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
| Too many steps | Reject the whole plan, use the default plan |
| Empty/unparseable LLM response | Caught one layer up (`src/llm.py`), use the default plan |

Each of these was independently triggered and verified during development, not just written
and hoped for.

---

## Failure cases observed

The assignment asks for one; two distinct ones turned up during testing, and both changed
the code, so both are worth recording honestly.

### 1. A tool that "succeeds" but returns nothing usable (the primary one)

`summarizer`'s output schema is `{"points": list[str]}`, with `points` defaulting to `[]` if
absent. Under real load, `gpt-oss-20b` intermittently returned a *structurally valid*,
schema-passing response of `{"points": []}` — despite being handed 12 real, substantial
search results moments earlier. Reproduced directly: identical model, prompt, and material;
one call empty, the next call (with `openai/gpt-oss-20b` again) with the same input, five
solid points.

This mattered downstream: `content_generator`'s `material_and_flags()` treats an empty
`points` list the same as a failed dependency ("degraded"). When every dependency looks
degraded, the generator correctly refuses per the "refuse rather than fabricate" rule — so
one flaky call was silently taking down runs that had perfectly good source material the
entire time. `src/llm.chat_json` only checked `result is None`; it never checked whether a
*structurally valid* result was actually empty.

**Fix:** `summarizer.py` (and, for the same reason, `generator.py`) now treat "succeeded but
empty" the same as "failed" and fall through to the deterministic fallback (raw result
titles for the summarizer; a plain topic digest for the generator). Confirmed with 3
consecutive full pipeline runs post-fix, zero false refusals; the same 3-run sequence
pre-fix false-refused on every run once this path was hit.

**The lesson:** a Pydantic schema constrains *shape*, not *usefulness*. "The call didn't
raise and the JSON validated" is not the same claim as "the call gave us something worth
using" — anywhere a field has a default, an LLM can satisfy the schema by doing nothing.

### 2. A second, related bug: refusing when refusal wasn't the planner's intent

Fixing (1) surfaced a second bug in the same area. `content_generator`'s refusal check was
originally `if not material.strip(): refuse`. That's correct when dependencies exist and all
of them failed — but a topic like *"why I left consulting"* gets a 2-step plan with **no
dependencies at all** (see above): the planner *deliberately* didn't wire any search in. With
the original check, `deps` is `{}`, `material` is `""`, and the generator refused
unconditionally — meaning every no-search plan the planner could ever produce would
immediately backfire. Fixed by distinguishing "no dependencies were given" (write from
general knowledge — appropriate for an opinion piece) from "dependencies existed and all
were degraded" (refuse, per the rule above): the check is now `if deps and not
material.strip()`.

Together these two are really one lesson: refusal logic needs to know not just *what
happened* but *what was supposed to happen* — an empty result means something different
depending on whether material was ever expected to exist.

### 3. "Healthy" is not "working" — the UI would have crashed in Docker

A later verification pass, driving the UI instead of only the API, found that
`streamlit run ui/app.py` puts the **script's** directory (`ui/`) on `sys.path` — not the
project root — and because `streamlit` is an installed console script, the working directory
isn't added either. So `from config import API_BASE_URL` raised `ModuleNotFoundError` in the
container.

What makes this worth recording is *why it was missed*: the earlier check confirmed
`/_stcore/health` returned `ok`, and concluded the UI worked. But **Streamlit only executes
the script when a browser session connects** — the health endpoint answers from the server
process, not the script. The container reported healthy while containing a UI that could not
start. Fixed with an explicit `sys.path.insert` of the project root (with the reason written
at the call site), verified by emulating the console-script `sys.path` inside the real image.

Two smaller issues from the same pass, both fixed: the image fallback ladder's timeouts
(15s + 12s) had to be retuned to fit inside `STEP_TIMEOUT_S` (45s), or a double network
hang would cancel the step before the guaranteed local card could render; and hand-unchecking
both search steps in the UI left `summarizer` orphaned, which caused a refusal the planner
itself would never produce (fixed by a validator pass that prunes transform-only steps left
with no dependencies — see `requires_deps` in `src/tools/__init__.py`).

---

## Design decisions

- **Orchestration:** LangChain primitives (`ChatGroq`, `@tool`-shaped tool functions) plus a
  hand-written async DAG executor, not LangGraph. LangGraph's `StateGraph` compiles its
  edges at build time; this plan is generated fresh per request, so a `StateGraph` would
  either be rebuilt every request for no benefit, or hard-code edges and let the planner only
  toggle nodes — which would quietly make the `/plan` endpoint's DAG decorative.
- **Search:** Tavily (`web_search`, general topic + advanced depth — `topic="news"` was
  tried and measured worse: 1 result at score 0.15 on this project's own example query,
  vs. 6 results at up to 0.65 for `general`) and DDGS (`social_search`, site-restricted to
  x.com/reddit.com/news.ycombinator.com). The assignment names `x_search_api`; the real X
  API v2 search endpoint is paid (~$200/mo) and not used here — DDGS is the honest,
  keyless substitute, restricted to platforms where people actually discuss things.
- **Image generation:** three-tier fallback — Cloudflare Workers AI (`flux-1-schnell`, 10k
  free neurons/day, no card, confirmed working live) → Pollinations (keyless, no signup) →
  a locally rendered typographic card (PIL, no network at all). The demo can never show a
  broken image, even offline. A cheap "art-director" LLM call turns the topic (or, if the
  planner wired it that way, the finished post) into a real visual brief first — a bare
  topic string is a bad diffusion prompt.
- **Structured output:** `ChatGroq.with_structured_output(method="json_schema")` — Groq's
  own server-side schema enforcement, verified working on both `gpt-oss-20b` and
  `gpt-oss-120b`. Not `method="json_mode"` (tried first, removed — see below) and not the
  default `method="function_calling"`, which has an open incompatibility with `gpt-oss-*`
  models on Groq ([langchain#34155](https://github.com/langchain-ai/langchain/issues/34155)).
  **A real bug lived in the `json_mode` path during development**: it tells Groq "emit valid
  JSON" but — per LangChain's own docstring — never tells the model *what shape*, so the
  model invented its own field names (`{"post": "..."}` instead of `{"headline", "body",
  "hashtags"}`) and every call failed Pydantic validation. Switching to `json_schema` fixed
  it outright.
- **Editor:** critique → revise, not a single "improve this" rewrite, so the UI can show
  *what* the editor found plus a before/after of what actually changed — not just a
  different blob of text. The rewrite prompt bans introducing any statistic, percentage,
  company or product name absent from the draft: an earlier, looser wording ("do not invent
  new claims") let a thin draft become *"Boost Backend Productivity by 30% with Serverless
  Automation"* — a fabricated figure and a technology never mentioned. A model told to fix
  "too vague" will manufacture specifics unless concretely forbidden.
- **UI flow:** two calls (`/plan` then `/execute`), with the plan rendered as an editable
  checklist between them — the plan is visible, and changeable, before anything expensive
  runs. `/execute/stream` (SSE) additionally ticks each tool live as it finishes, which is
  the clearest way to *watch* the parallelism claim rather than take it on faith.
- **Refusal:** refuse only if every dependency is degraded, not on any single tool failure —
  see the failure cases above for how this interacted with the planner's own decisions.

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

`/execute/stream` is the same contract over Server-Sent Events (`event: start|done|planned|result`,
`data:` a JSON payload) — used by the Streamlit UI for live progress, not required by the
assignment on its own.

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

> The `src.tools.*` commands print a `RuntimeWarning` from `runpy` about the module being
> "found in sys.modules" — harmless and stderr-only. It happens because `src/tools/__init__.py`
> eagerly imports every tool to build the registry, so running one as `__main__` imports it a
> second time. The JSON on stdout is unaffected, and nothing that actually serves traffic
> (uvicorn, streamlit) goes through this path. Silencing it would mean making the registry
> lazy and changing every consumer of `TOOLS` — not a trade worth making for a warning on a
> convenience command.

---

## Repo layout

```
config.py                 every knob, each with a comment saying WHY
requirements.txt          pinned versions, each with a reason
Dockerfile                one image, shared by the api and ui services
docker-compose.yml         api (:8000) + ui (:8502)
docker-compose.override.yml   dev-only bind mount for live reload (auto-loaded)
api/
  main.py                 /health /plan /execute /execute/stream /revise
  schemas.py               request/response models
src/
  llm.py                  the ONLY place an LLM is called; chat_json() + schema enforcement
  plan_schema.py           Step/Plan models + the default plan
  planner.py               Planner Agent
  validator.py             deterministic repair of a planner's DAG
  executor.py               per-step-readiness async DAG executor + trace
  types.py                 StepResult, shared between tools and the executor
  tools/
    web_search.py           Tavily
    social_search.py        DDGS
    summarizer.py            condenses raw search results into key points
    generator.py             Generator Agent
    editor.py                Editor Agent
    image_gen.py             art director + 3-tier image fallback
ui/app.py                  Streamlit UI
```

## Limitations

- Cloudflare's free tier is 10,000 neurons/day shared across every model call on the
  account — heavy demo use can exhaust it; Pollinations and the local card exist for exactly
  that reason.
- DDGS is an unofficial wrapper around DuckDuckGo's HTML; it can break when DuckDuckGo
  changes markup, independent of anything in this codebase.
- The planner's tool-argument tuning (e.g. `time_range`) is only as good as the one-shot LLM
  call that produces it — the validator repairs *structural* mistakes, not judgment calls
  that are merely suboptimal rather than invalid.
