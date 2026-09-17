"""social_search -- DDGS, restricted to platforms where people discuss a topic
informally. The assignment's example plan names "x_search_api", but the real
X API v2 search endpoint is paid (~$200/mo) and out of reach here.
This is the honest substitute: no key, no signup, and it
genuinely returns a different flavour of material than web_search -- opinion
and discussion threads rather than articles -- which is the whole point of
running it as a second, independent tool in the same parallel wave.
"""

from __future__ import annotations

import asyncio

from ddgs import DDGS
from pydantic import BaseModel, Field

from config import SOCIAL_MAX_RESULTS, SOCIAL_SITES, SOCIAL_TIMELIMIT
from src.tools._util import tool_fail, tool_ok
from src.types import StepResult

NAME = "social_search"
DESCRIPTION = (
    "Search social/discussion platforms (X, Reddit, Hacker News) for how "
    "people are talking about a topic right now. Returns titles, URLs, and "
    "short snippets. This is a substitute for the paid X API."
)


class SocialSearchArgs(BaseModel):
    query: str | None = Field(
        default=None, description="override search query; defaults to the topic"
    )


def _search_sync(query: str) -> dict:
    site_filter = " OR ".join(f"site:{s}" for s in SOCIAL_SITES)
    full_query = f"{query} ({site_filter})"
    try:
        hits = DDGS().text(
            full_query,
            max_results=SOCIAL_MAX_RESULTS,
            timelimit=SOCIAL_TIMELIMIT,
        )
    except Exception as e:  # noqa: BLE001 -- DDGS is unofficial; it breaks in varied ways
        return tool_fail(f"DDGS request failed: {e}")

    results = [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]
    return tool_ok(query=full_query, sites=SOCIAL_SITES, results=results)


async def run(topic: str, args: dict, deps: dict[int, StepResult]) -> dict:
    parsed = SocialSearchArgs.model_validate(args or {})
    query = parsed.query or topic
    return await asyncio.to_thread(_search_sync, query)


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "GenAI agents for backend engineers"
    print(json.dumps(asyncio.run(run(q, {}, {})), indent=2)[:2000])
