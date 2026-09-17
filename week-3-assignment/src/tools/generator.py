"""content_generator -- the Generator Agent. Takes aggregated content and
writes a LinkedIn-style post: headline, body, hashtags.

Uses MODEL_SMART (gpt-oss-120b) -- this is the first of the two things the
user actually reads (the other is the editor's revision), so it gets the
stronger model, following the project's "fast model for decisions, smart model
for output" split (see config.py).
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from config import MAX_HASHTAGS, MODEL_SMART
from src import llm
from src.tools._util import material_and_flags, tool_ok
from src.types import StepResult

NAME = "content_generator"
DESCRIPTION = (
    "Turn aggregated source material into a LinkedIn-style post: a headline, "
    "a concise professional body, and relevant hashtags. Refuses if none of "
    "its dependencies produced usable material, rather than writing from the "
    "model's own unsourced knowledge."
)

SYSTEM = (
    "You write LinkedIn posts for a technical, professional audience. Be concise, "
    "concrete, and specific -- reference real facts from the source material rather "
    "than generic statements. No emoji spam, no clickbait, no exclamation-mark chains. "
    "The body should read like a practitioner sharing something they noticed, not an "
    "ad. 100-200 words for the body."
)


class GeneratedPost(BaseModel):
    headline: str = Field(description="a short, specific hook line, not a generic title")
    body: str = Field(description="the post body, 100-200 words, plain text with line breaks")
    hashtags: list[str] = Field(description="3-6 relevant hashtags, no leading #")


class GeneratorArgs(BaseModel):
    audience: str | None = Field(
        default=None, description="who the post should be written for, e.g. 'backend engineers'"
    )


def _fallback_post(topic: str, material: str) -> dict:
    """Deterministic fallback when the LLM call fails outright -- a plain
    digest of the material rather than nothing, or (for a no-search plan with
    no material to digest) just the topic itself."""
    lines = [l.lstrip("- ").strip() for l in material.split("\n") if l.strip()][:3] or [topic]
    body = f"Some notes on {topic}:\n\n" + "\n".join(f"• {l}" for l in lines)
    words = [w.strip(".,:;").lower() for w in topic.split() if len(w) > 3][:4]
    hashtags = [w.capitalize() for w in words] or ["GenAI"]
    return {
        "headline": f"Notes on {topic}",
        "body": body,
        "hashtags": hashtags[:MAX_HASHTAGS],
    }


async def run(topic: str, args: dict, deps: dict[int, StepResult]) -> dict:
    parsed = GeneratorArgs.model_validate(args or {})
    material, degraded = material_and_flags(deps)

    # These are two different situations that both leave `material` empty,
    # and only one of them should refuse:
    #   - deps is non-empty but every dependency came back degraded: search
    #     was attempted and found nothing usable. In that case,
    #     refuse rather than write an ungrounded "recent trends" post from
    #     the model's own (possibly stale, possibly wrong) training data.
    #   - deps is EMPTY: the planner deliberately gave content_generator no
    #     dependencies at all (e.g. a personal-opinion topic like "why I left
    #     consulting" doesn't benefit from web material -- see src/planner.py's
    #     system prompt). That is not a search failure; refusing here would
    #     make the planner's own considered "skip search" decision always
    #     backfire. This was a real bug caught during manual testing: the
    #     first version of this check refused on `not material.strip()`
    #     alone, which refused every no-search plan unconditionally.
    if deps and not material.strip():
        return tool_ok(
            refused=True,
            reason="no usable source material from any dependency",
            degraded_sources=degraded,
        )

    audience_line = f"Target audience: {parsed.audience}\n\n" if parsed.audience else ""
    if material.strip():
        user = f"Topic: {topic}\n\n{audience_line}Source material:\n{material}"
    else:
        user = (
            f"Topic: {topic}\n\n{audience_line}No external source material was provided -- "
            "this is a personal/opinion topic. Write from general knowledge and a first-person "
            "perspective rather than citing sources or claiming specific recent events."
        )

    # Offload the sync Groq call -- see summarizer.py's comment; this one
    # matters even more since MODEL_SMART calls run longer than MODEL_FAST ones.
    post = await asyncio.to_thread(llm.chat_json, SYSTEM, user, GeneratedPost, MODEL_SMART, 0.6, 700)
    # GeneratedPost's fields are all required (no defaults), so an empty
    # *list* like summarizer's can't slip through here -- but an empty
    # *string* headline/body still validates fine. Same "succeeded but
    # useless" class of failure (see summarizer.py for the live case that
    # motivated checking for it at all); guard it the same way.
    if post is None or not post.headline.strip() or not post.body.strip():
        payload = _fallback_post(topic, material)
        payload["_fallback"] = True
    else:
        payload = post.model_dump()
        payload["hashtags"] = payload["hashtags"][:MAX_HASHTAGS]

    payload["refused"] = False
    payload["degraded_sources"] = degraded
    return tool_ok(**payload)


if __name__ == "__main__":
    import asyncio
    import json
    import sys

    async def _demo():
        topic = " ".join(sys.argv[1:]) or "recent trends in GenAI agents for backend engineers"
        fake_dep = StepResult(
            step=1,
            tool="summarizer",
            ok=True,
            output={"points": [f"{topic} is a fast-moving area with new frameworks shipping monthly."]},
        )
        return await run(topic, {}, {1: fake_dep})

    print(json.dumps(asyncio.run(_demo()), indent=2))
