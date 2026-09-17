"""FastAPI backend agent orchestration. Deliverable endpoints: /plan and
/execute. /execute/stream is an addition on top -- SSE so the UI can tick
each tool green as it finishes -- but /execute alone
already satisfies the assignment's stated contract on its own with a plain
JSON response, and both endpoints work with no UI involved at all (curl
examples are in README.md).
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    ExecuteRequest,
    ExecuteResponse,
    PlanRequest,
    PlanResponse,
    ReviseRequest,
    ReviseResponse,
)
from src import executor, planner
from src.tools import editor as editor_tool
from src.plan_schema import Plan
from src.types import StepResult
from src.validator import validate_and_repair

app = FastAPI(title="Agentic LinkedIn Content Curator")

# The UI is a separate origin (its own container/port) in the compose setup.
# This is a local training-assignment API, not a public one, so a wide-open
# CORS policy is a reasonable simplification -- narrowing it to the UI's
# actual origin would be the first change before this touched real users.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/plan", response_model=PlanResponse)
async def create_plan(req: PlanRequest) -> PlanResponse:
    # planner.plan_and_validate makes a blocking Groq call; keep the event
    # loop free for any concurrent request the same way the tools do.
    final_plan, notes, used_fallback = await asyncio.to_thread(planner.plan_and_validate, req.topic)
    return PlanResponse(plan=final_plan, repair_notes=notes, used_llm_fallback=used_fallback)


async def _resolve_plan(req: ExecuteRequest) -> tuple[Plan, list[str], bool]:
    if req.plan is not None:
        # The UI already showed this plan and let the user edit it -- planning
        # again here would silently discard those edits. But a user edit
        # (e.g. unchecking a step whose id another step's depends_on still
        # names) can introduce exactly the same structural problems an LLM
        # plan can, so it goes through the same deterministic repair rather
        # than a second, parallel set of edge-case handling.
        final, notes = validate_and_repair(req.plan)
        return final, notes, False
    return await asyncio.to_thread(planner.plan_and_validate, req.topic)


@app.post("/execute", response_model=ExecuteResponse)
async def execute_plan(req: ExecuteRequest) -> ExecuteResponse:
    final_plan, notes, used_fallback = await _resolve_plan(req)
    result = await executor.execute(req.topic, final_plan)
    result["trace"]["planner_notes"] = notes
    result["trace"]["used_llm_fallback"] = used_fallback
    return ExecuteResponse(**result)


def _sse(event_type: str, **data) -> dict:
    return {"event": event_type, "data": json.dumps(data)}


@app.post("/execute/stream")
async def execute_plan_stream(req: ExecuteRequest) -> EventSourceResponse:
    async def gen():
        yield _sse("planning")
        final_plan, notes, used_fallback = await _resolve_plan(req)
        yield _sse(
            "planned",
            plan=final_plan.model_dump(),
            repair_notes=notes,
            used_llm_fallback=used_fallback,
        )

        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: dict) -> None:
            queue.put_nowait(event)

        async def _run() -> None:
            result = await executor.execute(req.topic, final_plan, emit=emit)
            result["trace"]["planner_notes"] = notes
            result["trace"]["used_llm_fallback"] = used_fallback
            await queue.put({"event": "result", **result})
            await queue.put(None)  # sentinel: nothing more is coming

        task = asyncio.create_task(_run())
        while True:
            event = await queue.get()
            if event is None:
                break
            event_type = event.pop("event")
            yield _sse(event_type, **event)
        await task  # propagate an exception if _run somehow raised (it shouldn't -- executor never does)

    return EventSourceResponse(gen())


@app.post("/revise", response_model=ReviseResponse)
async def revise(req: ReviseRequest) -> ReviseResponse:
    """Run the Editor Agent again over an existing post.

    The editor tool reads its input from a dependency rather than from its
    arguments (that is how every tool in the DAG works), so this hands it a
    synthetic content_generator StepResult instead of special-casing the tool
    for this endpoint. The tool itself is unchanged and unaware it is being
    called outside a plan -- which is the point of tools being independently
    callable.

    Note the editor critiques whatever it is given, so feeding it its own
    previous output is what makes repeated presses keep refining rather than
    re-deriving the same edit from the original draft.
    """
    started = time.perf_counter()
    # Strip the previous round's editor metadata: `issues` and `original`
    # describe the LAST revision, and passing them back in would have the
    # editor critiquing its own critique rather than the post.
    draft = {k: v for k, v in req.post.items() if k not in ("issues", "original")}
    synthetic_dep = StepResult(step=0, tool="content_generator", ok=True, output=draft)
    output = await editor_tool.run(req.topic, {}, {0: synthetic_dep})
    return ReviseResponse(post=output, ms=int((time.perf_counter() - started) * 1000))
