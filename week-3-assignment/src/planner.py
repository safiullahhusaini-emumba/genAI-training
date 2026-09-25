"""The Planner Agent (Core Brain). Analyzes the topic, decides which tools to
call and in what order, and expresses that as a DAG: a flat list of steps
whose `depends_on` field determines execution order. See README.md's
"Why a planner is actually needed" -- the point demonstrated here is that the
tool *set* changes with the topic (a personal-opinion topic needs no search
at all; a "recent trends" topic needs a tight time_range) and that's a
per-request decision, not something a hard-coded pipeline can express.
"""

from __future__ import annotations

from config import MAX_PLAN_STEPS, MODEL_FAST
from src import llm
from src.plan_schema import DEFAULT_PLAN, Plan
from src.tools import describe_tools, tool_names

SYSTEM = f"""You are the planning agent for a LinkedIn content-curation system. Given a
topic, decide which tools to call and in what order, expressed as a dependency graph.

Available tools:
{describe_tools()}

Rules:
- Every plan must end with content_generator, then content_editor depending on it.
- Every plan must include exactly one image_generator step, whatever the topic --
  every post needs an image. Skipping research never means skipping the image.
- image_generator normally has depends_on: [] (it starts immediately, in parallel
  with search, using just the topic) -- this is faster. Only make it depend on
  content_generator's step if the topic's visual angle genuinely can't be
  determined without the finished post (e.g. the post's angle is a surprising
  contrarian take that the raw topic string doesn't convey).
- If the topic is personal opinion, a personal story, or otherwise not something
  that benefits from web material (e.g. a career lesson or a personal take), you may skip
  web_search and social_search entirely and wire content_generator directly to
  nothing (depends_on: []) -- do not search just because tools exist. Still
  include image_generator.
- If the topic needs recent material, pick web_search's time_range based on how
  fast-moving the topic is: "month" or "week" for fast-moving tech trends,
  "year" or omitted for evergreen/how-to topics.
- Use step ids starting at 1, no gaps, no duplicates. depends_on must reference
  only step ids that exist earlier in your own plan.
- At most {MAX_PLAN_STEPS} steps.

Respond with a JSON object: {{"steps": [{{"step": int, "tool": str, "depends_on": [int], "args": {{}}}}]}}

Known tool names, exactly: {", ".join(tool_names())}
"""


def plan(topic: str) -> tuple[Plan, bool]:
    """Returns (plan, used_fallback). used_fallback is True iff the LLM's
    plan was unusable (empty response, unparseable JSON, wrong shape -- all
    already collapsed to None by src.llm.chat_json) and DEFAULT_PLAN was
    substituted -- surfaced in the trace so the debug panel can show when the
    planner actually did nothing."""
    result = llm.chat_json(
        SYSTEM,
        f"Topic: {topic}",
        schema=Plan,
        model=MODEL_FAST,
        temperature=0.2,
        max_tokens=800,
    )
    if result is None or not result.steps:
        return DEFAULT_PLAN, True
    return result, False


def plan_and_validate(topic: str) -> tuple[Plan, list[str], bool]:
    """The full planning path used by /plan and /execute: ask the LLM, then
    run its answer through the deterministic validator/repair (src.validator)
    before anything gets executed. Returns (final_plan, repair_notes,
    used_llm_fallback) -- both failure layers are surfaced separately because
    they're different claims: "the LLM gave us nothing" vs "the LLM gave us
    something, and it needed N structural repairs"."""
    from src.validator import validate_and_repair  # local import: validator imports this module's Plan/DEFAULT_PLAN, not the reverse, but keeping it lazy avoids any future ordering surprise

    raw, used_fallback = plan(topic)
    if used_fallback:
        return DEFAULT_PLAN, ["planner returned invalid/empty output; used the default plan"], True
    final, notes = validate_and_repair(raw)
    return final, notes, False


if __name__ == "__main__":
    import json
    import sys

    t = " ".join(sys.argv[1:]) or "recent trends in GenAI agents for backend engineers"
    final_plan, notes, used_fallback = plan_and_validate(t)
    print(json.dumps({"used_llm_fallback": used_fallback, "repair_notes": notes, **final_plan.model_dump()}, indent=2))
