"""summarizer -- the assignment's "Summarization Tool": condense raw articles
into key points. Sits between the two search tools and the Generator Agent in
the default plan, but is registered like any other tool so the planner is
free to wire it in differently (or skip it and point content_generator
straight at the raw search steps, as the assignment's own example plan does).
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from config import MODEL_FAST
from src import llm
from src.tools._util import tool_ok
from src.types import StepResult

NAME = "summarizer"
DESCRIPTION = (
    "Condense raw search results (from web_search and/or social_search) into "
    "a short list of key points a writer can work from."
)

SYSTEM = (
    "You summarize raw search results into key points for someone about to write "
    "a LinkedIn post. Extract concrete facts, named tools/companies/numbers, and "
    "notable opinions -- skip filler. Each point should stand alone as one sentence."
)


class Point(BaseModel):
    points: list[str] = Field(default_factory=list, description="one sentence per key point, 4-8 total")


class SummarizerArgs(BaseModel):
    max_points: int = Field(default=8, ge=1, le=15)


def _gather_material(deps: dict[int, StepResult]) -> tuple[str, list[str]]:
    """Pull raw search results out of whichever dependency steps produced
    them. Returns (formatted text for the prompt, list of source URLs)."""
    chunks: list[str] = []
    sources: list[str] = []
    for dep in deps.values():
        if not dep.ok or not dep.output:
            continue
        for r in dep.output.get("results", []) or []:
            title = r.get("title", "")
            body = r.get("content") or r.get("snippet") or ""
            url = r.get("url", "")
            if not (title or body):
                continue
            chunks.append(f"- {title}: {body}".strip())
            if url:
                sources.append(url)
    return "\n".join(chunks), sources


async def run(topic: str, args: dict, deps: dict[int, StepResult]) -> dict:
    parsed = SummarizerArgs.model_validate(args or {})
    material, sources = _gather_material(deps)

    if not material.strip():
        # Both search dependencies came back empty or failed. Not this tool's
        # job to refuse (that's the Generator's call -- it refuses only if
        # ALL search fails) -- just report honestly
        # that it had nothing to work with.
        return tool_ok(points=[], sources=[], note="no search material available to summarize")

    # llm.chat_json is a sync network call; offload it so it can't stall
    # other steps running concurrently in a different wave (e.g. image_gen's
    # Cloudflare call, which has no dependency on this step at all).
    result = await asyncio.to_thread(
        llm.chat_json,
        SYSTEM,
        f"Topic: {topic}\n\nRaw search results:\n{material}\n\n"
        f"Return at most {parsed.max_points} key points.",
        Point,
        MODEL_FAST,
        0.2,
        500,
    )
    if result is None or not result.points:
        # Deterministic fallback: use the raw result titles as points
        # rather than failing the whole step over one LLM call.
        #
        # The `not result.points` half was found live, not planned: Point's
        # `points` field has default_factory=list, so {"points": []} is a
        # structurally VALID response the schema can't forbid -- and under
        # real load gpt-oss-20b intermittently returns exactly that despite
        # being handed perfectly good search material (reproduced directly:
        # same model, same prompt, same material, one call empty, the next
        # not). Treating "succeeded but empty" the same as "failed" matters
        # downstream: content_generator's material_and_flags() treats an
        # empty points list as this dependency being degraded, and if every
        # dependency looks degraded, the generator refuses outright (D10) --
        # so a single flaky call here was silently taking down a run that had
        # perfectly good source material the whole time.
        points = [line.lstrip("- ") for line in material.split("\n") if line.strip()][: parsed.max_points]
        return tool_ok(points=points, sources=sources, note="LLM summarization failed; used raw titles")

    return tool_ok(points=result.points[: parsed.max_points], sources=sources)


if __name__ == "__main__":
    import asyncio
    import json
    import sys

    from src.tools.web_search import run as web_search_run

    async def _demo():
        topic = " ".join(sys.argv[1:]) or "recent trends in GenAI agents for backend engineers"
        search_out = await web_search_run(topic, {}, {})
        fake_dep = StepResult(step=1, tool="web_search", ok=search_out.get("ok", True), output=search_out)
        return await run(topic, {}, {1: fake_dep})

    print(json.dumps(asyncio.run(_demo()), indent=2))
