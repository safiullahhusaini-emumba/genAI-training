"""web_search -- Tavily. The general web-discovery tool: find recent,
relevant content on the topic.

Configuration for topic/search_depth is fixed in config.py, not exposed as a
planner argument, because measurement showed topic="news" starves technical
queries (1 result, score 0.15, on a representative technical topic) while
topic="general"+advanced does not. time_range IS a planner argument -- the
one axis where the right answer genuinely depends on what the topic is
asking for.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field
from tavily import TavilyClient

from config import (
    TAVILY_API_KEY,
    TAVILY_DEFAULT_TIME_RANGE,
    TAVILY_MAX_RESULTS,
    TAVILY_SEARCH_DEPTH,
    TAVILY_TOPIC,
)
from src.tools._util import tool_fail, tool_ok
from src.types import StepResult

NAME = "web_search"
DESCRIPTION = (
    "Search the public web for recent articles and discussion on a topic. "
    "Returns titles, URLs, a relevance score, and extracted page content."
)


class WebSearchArgs(BaseModel):
    query: str | None = Field(
        default=None, description="override search query; defaults to the topic"
    )
    time_range: str | None = Field(
        default=None,
        description=(
            "one of day/week/month/year, or omit for no recency filter. Use "
            "'month' or tighter for fast-moving topics ('recent trends in X'); "
            "omit or use 'year' for evergreen/how-to topics where a wider net "
            "returns better material without losing relevance."
        ),
    )


def _search_sync(query: str, time_range: str | None) -> dict:
    if not TAVILY_API_KEY:
        return tool_fail("TAVILY_API_KEY is not set. Copy .env.example to .env.")
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = client.search(
            query=query,
            topic=TAVILY_TOPIC,
            search_depth=TAVILY_SEARCH_DEPTH,
            time_range=time_range,
            max_results=TAVILY_MAX_RESULTS,
            include_raw_content=False,
        )
    except Exception as e:  # noqa: BLE001 -- network/API errors of any shape land here
        return tool_fail(f"Tavily request failed: {e}")

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
            "published_date": r.get("published_date"),
        }
        for r in resp.get("results", [])
    ]
    return tool_ok(query=query, time_range=time_range, results=results)


async def run(topic: str, args: dict, deps: dict[int, StepResult]) -> dict:
    parsed = WebSearchArgs.model_validate(args or {})
    query = parsed.query or topic
    time_range = parsed.time_range or TAVILY_DEFAULT_TIME_RANGE
    # tavily-python's client is sync; keep the executor's event loop free.
    return await asyncio.to_thread(_search_sync, query, time_range)


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "recent trends in GenAI agents for backend engineers"
    print(json.dumps(asyncio.run(run(q, {}, {})), indent=2)[:2000])
