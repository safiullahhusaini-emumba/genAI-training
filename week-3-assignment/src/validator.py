"""Deterministic repair of whatever the Planner Agent returns. An LLM-written
DAG fails in a small, repeatable set of ways -- this module is the
non-negotiable safety net between "the planner said so" and "the executor
runs it". Every repair is recorded as a note that
ends up in the trace, so the debug panel shows exactly what the planner got
wrong on a given run (this is also where the README's required "one failure
case you observed" comes from).

Failure modes handled here, each independently triggerable for testing:
  - hallucinated tool name              -> drop the step
  - dependency on a nonexistent step    -> drop the dependency
  - duplicate step ids                  -> renumber (first occurrence wins)
  - a dependency cycle                  -> reject the whole plan, use DEFAULT_PLAN
  - missing content_generator/_editor   -> inject them at the end
  - too many steps                      -> reject the whole plan, use DEFAULT_PLAN

Two more failure modes -- syntactically invalid JSON, and an empty/garbage
response -- never reach this module at all: src.llm.chat_json already returns
None for those, and src.planner substitutes DEFAULT_PLAN before this code runs.
"""

from __future__ import annotations

from config import MAX_PLAN_STEPS
from src.plan_schema import DEFAULT_PLAN, EDITOR_TOOL, GENERATOR_TOOL, Plan, Step
from src.tools import TOOLS, tool_names
from src.tools.image_gen import NAME as IMAGE_TOOL


def validate_and_repair(raw: Plan) -> tuple[Plan, list[str]]:
    notes: list[str] = []
    known = set(tool_names())

    filtered = [s for s in raw.steps if s.tool in known]
    for s in raw.steps:
        if s.tool not in known:
            notes.append(f"dropped step {s.step} ('{s.tool}'): not a known tool")

    if not filtered:
        notes.append("no valid steps remained after filtering; using the default plan")
        return DEFAULT_PLAN, notes

    filtered = _prune_orphaned_transforms(filtered, notes)

    if not filtered:
        notes.append("no valid steps remained after pruning; using the default plan")
        return DEFAULT_PLAN, notes

    # Renumber sequentially in list order. id_map resolves an ORIGINAL step id
    # to its new sequential id, first occurrence wins, so a duplicate id
    # (planner mistake) can't create ambiguity about which step a later
    # dependency meant -- it always means the first one seen with that id.
    id_map: dict[int, int] = {}
    for i, s in enumerate(filtered, start=1):
        if s.step in id_map:
            notes.append(f"duplicate step id {s.step} ('{s.tool}') renumbered to {i}")
        else:
            id_map[s.step] = i

    new_steps: list[Step] = []
    for i, s in enumerate(filtered, start=1):
        remapped = []
        for d in s.depends_on:
            if d in id_map:
                remapped.append(id_map[d])
            else:
                # `i` is the new (renumbered) id, `d` the planner's original
                # id -- say so, or a note like "step 2 dropped dependency on
                # missing step 2" reads like a nonsensical self-reference.
                notes.append(
                    f"step {i} ('{s.tool}'): dropped dependency on original step {d}, "
                    "which is not in the plan"
                )
        deps = sorted({x for x in remapped if x != i})  # x != i: a step can't depend on itself post-renumber
        new_steps.append(Step(step=i, tool=s.tool, depends_on=deps, args=s.args))

    graph = {s.step: s.depends_on for s in new_steps}
    if _has_cycle(graph):
        notes.append("dependency cycle detected; using the default plan")
        return DEFAULT_PLAN, notes

    # Every plan must terminate in content_generator -> content_editor,
    # regardless of what the planner decided to search with.
    tool_to_id: dict[str, int] = {}
    for s in new_steps:
        tool_to_id.setdefault(s.tool, s.step)

    if GENERATOR_TOOL not in tool_to_id:
        gen_id = len(new_steps) + 1
        upstream = [s.step for s in new_steps if s.tool != IMAGE_TOOL]
        new_steps.append(Step(step=gen_id, tool=GENERATOR_TOOL, depends_on=upstream, args={}))
        tool_to_id[GENERATOR_TOOL] = gen_id
        notes.append("planner omitted content_generator; injected it depending on all non-image steps")

    if EDITOR_TOOL not in tool_to_id:
        ed_id = len(new_steps) + 1
        new_steps.append(Step(step=ed_id, tool=EDITOR_TOOL, depends_on=[tool_to_id[GENERATOR_TOOL]], args={}))
        notes.append("planner omitted content_editor; injected it depending on content_generator")

    if len(new_steps) > MAX_PLAN_STEPS:
        notes.append(f"plan has {len(new_steps)} steps, over the {MAX_PLAN_STEPS} cap; using the default plan")
        return DEFAULT_PLAN, notes

    return Plan(steps=new_steps), notes


def _prune_orphaned_transforms(steps: list[Step], notes: list[str]) -> list[Step]:
    """Drop transform-only steps (TOOLS[...].requires_deps) that have been left
    with no surviving dependencies, repeatedly until stable.

    This exists because of a real UX failure: in the Streamlit plan editor, a
    user unchecking both search steps leaves `summarizer` with an empty
    depends_on. It then runs, has nothing to summarize, and returns an empty
    points list -- which content_generator reads as "this dependency is
    degraded" and, since it is then the ONLY dependency, refuses the whole run
    (D10). The user asked for "write this without web research", which the
    system handles fine when the *planner* decides it (a no-search plan wires
    content_generator to nothing at all), and got a refusal instead.

    Pruning the dead step makes the hand-edited plan converge on exactly the
    shape the planner would have produced for the same intent. The loop (rather
    than a single pass) handles a chain of transforms feeding transforms.
    """
    surviving = list(steps)
    while True:
        alive_ids = {s.step for s in surviving}
        orphaned = [
            s
            for s in surviving
            if TOOLS[s.tool].requires_deps and not [d for d in s.depends_on if d in alive_ids]
        ]
        if not orphaned:
            return surviving
        for s in orphaned:
            notes.append(f"dropped step {s.step} ('{s.tool}'): nothing left for it to work on")
        dropped = {s.step for s in orphaned}
        surviving = [s for s in surviving if s.step not in dropped]


def _has_cycle(graph: dict[int, list[int]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(graph, WHITE)

    def dfs(node: int) -> bool:
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            state = color.get(neighbor, WHITE)
            if state == GRAY or (state == WHITE and dfs(neighbor)):
                return True
        color[node] = BLACK
        return False

    return any(color[n] == WHITE and dfs(n) for n in graph)
