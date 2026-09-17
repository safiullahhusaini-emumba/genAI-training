"""Central knobs. Everything tunable lives here so the pipeline files stay boring.

House rule for this file: every number carries a comment saying *why* it is that
value, not just what it is. A bare constant is a decision nobody can revisit.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

# --- Groq (LLM) ----------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Deliberate split: the fast model handles every intermediate decision
# (planning, summarizing, art-directing, critiquing), and the smart model is
# reserved for the two things the user actually reads -- the post and the
# revised post. Cheaper and quicker where quality barely shows, stronger where
# it is the whole output.
MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b")   # planner, summarizer, art director, editor's critique
MODEL_SMART = os.getenv("GROQ_MODEL_SMART", "openai/gpt-oss-120b")  # generator, editor's rewrite

# gpt-oss models on Groq are reasoning models; left at Groq's default they burn
# 250+ reasoning tokens before the answer starts, which truncates short-budget
# calls. langchain-groq 1.1.3 exposes reasoning_effort/reasoning_format as
# native ChatGroq fields, so no extra_body passthrough is needed to set them.
REASONING_EFFORT = "low"
REASONING_FORMAT = "hidden"

# --- Tavily (web search) --------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Measured against the assignment's own example query before picking these.
# topic="news" restricts to news-wire content and returned 1 result at score
# 0.15 for "recent trends in GenAI agents for backend engineers" -- a
# technical-trend query is not news-wire. topic="general"
# + search_depth="advanced" scored 0.60 max / 1071 chars avg on the same query,
# a large enough gap that fixing it here beats leaving it as a planner choice.
TAVILY_TOPIC = "general"
TAVILY_SEARCH_DEPTH = "advanced"
# time_range ("day"/"week"/"month"/"year") is NOT fixed here -- it is a
# planner-chosen tool argument (src/planner.py), because the right recency
# window genuinely depends on the topic: "recent trends" wants "month",
# "fundamentals of X" wants "year" or no filter at all. This default only
# applies if the planner omits it.
TAVILY_DEFAULT_TIME_RANGE = "month"
TAVILY_MAX_RESULTS = 6

# --- DDGS (social/discussion search) --------------------------------------------
# The assignment's example plan names "x_search_api", but the real X API v2
# search endpoint is paid (~$200/mo) -- not usable here. This is the honest
# substitute: DDGS needs no key at all, restricted to platforms where people
# actually discuss a topic in public, informal terms search engines rank differently.
SOCIAL_SITES = ["x.com", "reddit.com", "news.ycombinator.com"]
SOCIAL_MAX_RESULTS = 6
SOCIAL_TIMELIMIT = "m"  # DDGS: d/w/m/y -- discussion threads age out faster than articles

# --- Cloudflare Workers AI (image generation) -----------------------------------
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
CLOUDFLARE_STEPS = 4       # schnell is a distilled few-step model; max allowed is 8, 4 is the documented sweet spot
# Measured live latency was ~3-4s. 15s is 4x headroom, and is deliberately NOT
# larger: see the tier-budget note on STEP_TIMEOUT_S below -- the three image
# tiers have to fit inside one step's timeout or the last one never runs.
CLOUDFLARE_TIMEOUT = 15

# Fallback #2: keyless, no signup, but rate-limited (~1 req/15s anonymously)
# and carries no uptime guarantee -- fine as a fallback, risky as a primary.
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"
POLLINATIONS_TIMEOUT = 12

# Fallback #3: rendered locally with PIL, no network at all. Guarantees the
# demo can never show a broken image even if both network providers are down.
IMAGE_SIZE = (1024, 1024)

# --- planner / executor ---------------------------------------------------------
MAX_PLAN_STEPS = 8      # generous headroom above the 5-step example; a runaway planner still gets capped
# Per-tool wall clock, enforced by the executor's asyncio.wait_for.
#
# TIER BUDGET (the non-obvious constraint): image_generator's whole point is
# that it ALWAYS returns something -- Cloudflare, else Pollinations, else a
# locally rendered card that needs no network. That guarantee only holds if
# every tier fits inside this timeout, because wait_for cancels the step
# wherever it happens to be. Worst case is art-brief LLM (~5s) +
# CLOUDFLARE_TIMEOUT (15) + POLLINATIONS_TIMEOUT (12) = ~32s, leaving ~13s of
# margin before the local card would be pre-empted. Raise those two and this
# has to rise with them, or the "can never show a broken image" claim quietly
# becomes false in exactly the case it exists for.
STEP_TIMEOUT_S = 45
MAX_HASHTAGS = 6

# --- FastAPI / networking --------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
# Inside docker-compose the UI reaches the API by service name; on the host
# (or a bare `streamlit run`) it's localhost. Compose overrides this env var
# rather than the value being guessed at runtime.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Set to "1" by docker-compose.yml. There is no reliable way to detect this
# from inside the process (/.dockerenv is a Docker implementation detail,
# absent under Podman), so compose says so explicitly rather than guessing.
# Used only to print the right fix command when the UI can't reach the API --
# telling someone to run `docker compose up` when they're already inside the
# container, or `uvicorn` when they're not, sends them down a dead end.
IN_CONTAINER = os.getenv("CURATOR_IN_CONTAINER") == "1"

API_FIX_CMD = (
    "docker compose up -d\ndocker compose ps      # wait for both (healthy)"
    if IN_CONTAINER
    else "uvicorn api.main:app --reload      # from the project root, in another terminal"
)
