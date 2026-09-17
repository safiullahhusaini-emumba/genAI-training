"""content_editor -- the Editor Agent. Critiques the generator's draft, then
rewrites it addressing the critique.

Split into two calls, each on the model that fits:
the critique is a *decision* (what's wrong) so it uses MODEL_FAST; the
rewrite is *what the user reads*, the second and final thing that is, so it
uses MODEL_SMART. This also gives the debug panel something to show that a
single "improve this" rewrite couldn't: a numbered list of what changed and why.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from config import MAX_HASHTAGS, MODEL_FAST, MODEL_SMART
from src import llm
from src.tools._util import find_dep, tool_ok
from src.types import StepResult

NAME = "content_editor"
DESCRIPTION = (
    "Critique a generated LinkedIn post for clarity, tone, and structure, then "
    "produce a revised version. Passes through unchanged if the generator refused."
)

CRITIQUE_SYSTEM = (
    "You are an editor reviewing a LinkedIn post before publication. List concrete, "
    "actionable issues -- weak hooks, vague claims, generic hashtags, tone that doesn't "
    "fit a professional technical audience, structural problems. 2-5 issues. If the post "
    "is already strong, say so with one item noting what to preserve."
)
REVISE_SYSTEM = (
    "You are an editor rewriting a LinkedIn post to address a list of issues, while "
    "preserving every concrete fact and source detail from the original. "
    "You may only rephrase, restructure, and cut. You may NOT introduce any "
    "statistic, percentage, number, company name, product name, or factual claim "
    "that does not already appear in the post you were given -- if an issue asks "
    "for specifics the post does not contain, address it by sharpening what IS "
    "there, not by inventing details. Keep it 100-200 words, professional, no emoji spam."
)
# The explicit ban matters: the original wording was just "do not invent new
# claims", and a thin draft ("Lots of new frameworks this year") was rewritten
# into "Boost Backend Productivity by 30% with Serverless Automation" -- a
# fabricated statistic and a technology never mentioned. A model asked to fix
# "too vague" will manufacture specifics unless told, concretely, that it may
# not. This matters most on /revise, where repeated passes are grounded only in
# the previous pass's text rather than in the original search results.


class Critique(BaseModel):
    issues: list[str] = Field(description="2-5 concrete, actionable issues found in the draft")


class RevisedPost(BaseModel):
    headline: str
    body: str
    hashtags: list[str]


class EditorArgs(BaseModel):
    pass  # no tunables yet; kept as a real model so the schema story stays consistent


def _post_text(post: dict) -> str:
    tags = " ".join(f"#{h}" for h in post.get("hashtags", []))
    return f"Headline: {post.get('headline', '')}\n\nBody:\n{post.get('body', '')}\n\nHashtags: {tags}"


async def run(topic: str, args: dict, deps: dict[int, StepResult]) -> dict:
    EditorArgs.model_validate(args or {})
    gen_dep = find_dep(deps, "content_generator")
    if gen_dep is None or not gen_dep.ok or not gen_dep.output:
        return tool_ok(
            refused=True,
            reason="no content_generator output to edit",
            issues=[],
        )

    draft = gen_dep.output
    if draft.get("refused"):
        # Nothing to edit -- pass the refusal straight through rather than
        # spending an LLM call polishing a post that doesn't exist.
        return tool_ok(**draft, issues=[])

    # Both are sync Groq calls; offload so a slow critique/revise pass can't
    # stall an unrelated step still running elsewhere in the DAG (e.g. this
    # plan's own image_generator, which has no dependency on the editor).
    critique = await asyncio.to_thread(
        llm.chat_json, CRITIQUE_SYSTEM, f"Topic: {topic}\n\n{_post_text(draft)}", Critique, MODEL_FAST, 0.3, 300
    )
    if critique:
        # The model sometimes returns 5 points crammed into one list element
        # (newline-separated) instead of 5 elements -- a JSON *shape* the
        # schema can't forbid, only a content habit. Split defensively so the
        # debug panel shows one issue per card instead of one wall of text.
        issues = [line.strip("-• ") for item in critique.issues for line in item.split("\n") if line.strip()]
    else:
        issues = ["automatic critique unavailable; applied a light pass only"]

    revised = await asyncio.to_thread(
        llm.chat_json,
        REVISE_SYSTEM,
        f"Topic: {topic}\n\nOriginal post:\n{_post_text(draft)}\n\n"
        f"Issues to address:\n" + "\n".join(f"- {i}" for i in issues),
        RevisedPost,
        MODEL_SMART,
        0.4,
        700,
    )

    if revised is None:
        # Deterministic fallback: the generator's draft
        # is a perfectly usable post on its own -- ship it unedited rather
        # than fail the whole run because one rewrite call didn't come back.
        final = {"headline": draft["headline"], "body": draft["body"], "hashtags": draft["hashtags"]}
        issues.append("revision call failed; returned the generator's draft unchanged")
    else:
        final = revised.model_dump()
        final["hashtags"] = final["hashtags"][:MAX_HASHTAGS]

    return tool_ok(
        **final,
        issues=issues,
        original=draft,
        refused=False,
        degraded_sources=draft.get("degraded_sources", []),
    )


if __name__ == "__main__":
    import asyncio
    import json
    import sys

    async def _demo():
        topic = " ".join(sys.argv[1:]) or "recent trends in GenAI agents for backend engineers"
        fake_gen = StepResult(
            step=1,
            tool="content_generator",
            ok=True,
            output={
                "headline": "AI agents are changing backend work",
                "body": "Lots of new frameworks this year.",
                "hashtags": ["AI", "Backend"],
                "refused": False,
                "degraded_sources": [],
            },
        )
        return await run(topic, {}, {1: fake_gen})

    print(json.dumps(asyncio.run(_demo()), indent=2))
