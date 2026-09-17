"""The DAG executor: takes a validated Plan and actually runs it.

Per-step readiness scheduling, not level-based waves.
One asyncio.Task is created per step; each awaits only its OWN dependency
tasks, not a synchronized barrier for a whole "level". A level-based
scheduler -- group steps into topological levels, gather() each level, wait
for the whole level before the next -- introduces false dependencies: in the
default plan, summarizer (depends_on [1,2]) shares a level with
image_generator (depends_on []), so a level scheduler would make summarizer
wait for the slower of the two even though it has no relationship to the
image. Per-step readiness starts summarizer the instant steps 1 and 2 finish,
regardless of what else is still in flight. `wave` is still recorded in the
trace, but purely as a topological-level label for the UI's Gantt grouping --
it plays no part in scheduling.

A tool is never allowed to kill the whole run: every exception is caught here
and turned into a failed StepResult, so one dead API can't take down steps
that don't depend on it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from config import STEP_TIMEOUT_S
from src.plan_schema import Plan, Step
from src.tools import TOOLS
from src.types import StepResult

EventCallback = Callable[[dict], None]


def _topological_order(steps: list[Step]) -> list[Step]:
    """The planner can emit steps in any order; tasks must be CREATED in an
    order where every dependency's asyncio.Task already exists before the
    dependent step is scheduled. src.validator already guarantees the graph
    is acyclic, so plain DFS post-order suffices here."""
    by_id = {s.step: s for s in steps}
    ordered: list[Step] = []
    seen: set[int] = set()

    def visit(s: Step) -> None:
        if s.step in seen:
            return
        seen.add(s.step)
        for d in s.depends_on:
            if d in by_id:
                visit(by_id[d])
        ordered.append(s)

    for s in steps:
        visit(s)
    return ordered


def _compute_waves(steps: list[Step]) -> dict[int, int]:
    """Display-only topological level: 0 for a step with no deps, otherwise
    one more than the deepest dependency. Used purely so the UI can group
    steps into a Gantt strip -- the executor itself never waits on a "wave"."""
    by_id = {s.step: s for s in steps}
    memo: dict[int, int] = {}

    def level(step_id: int) -> int:
        if step_id in memo:
            return memo[step_id]
        deps = [d for d in by_id[step_id].depends_on if d in by_id]
        memo[step_id] = 0 if not deps else 1 + max(level(d) for d in deps)
        return memo[step_id]

    return {s.step: level(s.step) for s in steps}


async def _run_step(
    step: Step, topic: str, tasks: dict[int, asyncio.Task], emit: EventCallback | None
) -> StepResult:
    if step.depends_on:
        await asyncio.gather(*(tasks[d] for d in step.depends_on))
    deps = {d: tasks[d].result() for d in step.depends_on}

    started = time.perf_counter()
    if emit:
        emit({"event": "start", "step": step.step, "tool": step.tool})

    tool = TOOLS.get(step.tool)
    if tool is None:
        # Shouldn't happen -- src.validator already drops unknown tool names
        # -- but a hand-written plan (used in tests, or POSTed directly to
        # /execute) can bypass that, so this stays a reported failure rather
        # than a KeyError that would take the whole run down with it.
        result = StepResult(
            step.step, step.tool, ok=False, error=f"unknown tool '{step.tool}'",
            started_at=started, ended_at=time.perf_counter(),
        )
    else:
        try:
            output = await asyncio.wait_for(tool.fn(topic, step.args, deps), timeout=STEP_TIMEOUT_S)
            result = StepResult(
                step.step, step.tool, ok=bool(output.get("ok", True)), output=output,
                error=output.get("error"), started_at=started, ended_at=time.perf_counter(),
            )
        except Exception as e:  # noqa: BLE001 -- a tool must never be able to kill the whole run
            result = StepResult(
                step.step, step.tool, ok=False, error=f"{type(e).__name__}: {e}",
                started_at=started, ended_at=time.perf_counter(),
            )

    if emit:
        emit({"event": "done", "step": step.step, "tool": step.tool, "ok": result.ok, "ms": result.ms})
    return result


async def execute(topic: str, plan: Plan, emit: EventCallback | None = None) -> dict:
    t0 = time.perf_counter()
    ordered = _topological_order(plan.steps)
    waves = _compute_waves(plan.steps)

    tasks: dict[int, asyncio.Task] = {}
    for step in ordered:
        tasks[step.step] = asyncio.create_task(_run_step(step, topic, tasks, emit))

    results = await asyncio.gather(*tasks.values())
    by_step = {r.step: r for r in results}

    gen = next((r for r in results if r.tool == "content_generator"), None)
    editor = next((r for r in results if r.tool == "content_editor"), None)
    image = next((r for r in results if r.tool == "image_generator"), None)

    # The editor is the last word on the post if it ran and produced
    # something; otherwise fall back to the generator's draft (e.g. the
    # planner's own plan omitted an editor step, or the editor step failed
    # outright) rather than returning nothing when a perfectly good draft exists.
    final = None
    if editor and editor.ok and editor.output:
        final = editor.output
    elif gen and gen.ok and gen.output:
        final = gen.output

    if final and final.get("refused"):
        outcome = "refused"
    elif final:
        outcome = "answered"
    else:
        outcome = "failed"

    steps_trace = []
    for s in ordered:
        r = by_step[s.step]
        steps_trace.append(
            {
                **r.to_trace(),
                "wave": waves[s.step],
                "depends_on": s.depends_on,
                "start_ms": int((r.started_at - t0) * 1000),
                "end_ms": int((r.ended_at - t0) * 1000),
            }
        )

    trace = {
        "topic": topic,
        "plan": plan.model_dump(),
        "steps": steps_trace,
        "outcome": outcome,
        "total_ms": int((time.perf_counter() - t0) * 1000),
    }

    return {
        "post": final,
        "image": image.output if (image and image.ok) else None,
        "trace": trace,
    }
