"""Small helpers shared by every tool module. Split out from __init__.py so
tool modules can import these without triggering the registry's import of
every tool (which would be a circular import)."""

from __future__ import annotations

from src.types import StepResult


def tool_ok(**payload) -> dict:
    """Every tool returns a dict with an explicit "ok" key -- this is the
    "structured output" half of the assignment's "each tool must ... return
    structured output" requirement, and it's how the executor tells a tool's
    own considered decision (e.g. "refusing, no source material") apart from
    an unhandled exception, without inspecting error strings."""
    payload.setdefault("ok", True)
    return payload


def tool_fail(error: str, **extra) -> dict:
    return {"ok": False, "error": error, **extra}


def find_dep(deps: dict[int, StepResult], tool_name: str) -> StepResult | None:
    """The plan graph is addressed by step id, but a tool usually cares about
    *which tool* produced a dependency's output, not its arbitrary step number
    -- e.g. the editor wants "whatever fed me content_generator's output",
    regardless of whether the planner numbered it step 4 or step 7."""
    for dep in deps.values():
        if dep.tool == tool_name:
            return dep
    return None


def material_and_flags(deps: dict[int, StepResult]) -> tuple[str, list[str]]:
    """Pull usable source text out of whatever dependencies fed a step,
    whichever shape they're in -- summarizer's {"points": [...]} or a raw
    search tool's {"results": [...]}. This lets content_generator sit
    downstream of either the default plan (summarizer in between) or the
    assignment's own literal example plan (generator wired straight to the
    two search steps) without caring which.

    Returns (material_text, degraded_tool_names). A dependency counts as
    "degraded" if it failed outright OR succeeded with nothing usable --
    both are the same problem from the generator's point of view. The rule
    downstream is: refuse only when EVERY dependency is degraded.
    """
    chunks: list[str] = []
    degraded: list[str] = []
    for dep in deps.values():
        out = dep.output or {}
        if not dep.ok:
            degraded.append(dep.tool)
        elif out.get("points"):
            chunks.append("\n".join(f"- {p}" for p in out["points"]))
        elif out.get("results"):
            for r in out["results"]:
                text = r.get("content") or r.get("snippet") or ""
                if r.get("title") or text:
                    chunks.append(f"- {r.get('title', '')}: {text}".strip())
            if not out["results"]:
                degraded.append(dep.tool)
        else:
            degraded.append(dep.tool)
    return "\n\n".join(chunks), degraded
