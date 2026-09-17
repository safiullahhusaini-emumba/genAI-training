"""Request/response models for the two required endpoints. Kept separate from
src.plan_schema so the wire format (what a client POSTs/receives) can evolve
independently of the internal Plan representation the executor consumes."""

from __future__ import annotations

from pydantic import BaseModel

from src.plan_schema import Plan


class PlanRequest(BaseModel):
    topic: str


class PlanResponse(BaseModel):
    plan: Plan
    repair_notes: list[str]
    used_llm_fallback: bool


class ExecuteRequest(BaseModel):
    topic: str
    # Optional: the two-call UI flow calls /plan first, lets the user
    # edit the DAG, then POSTs that exact plan here. If omitted, /execute
    # plans internally -- both endpoints stay independently callable, which
    # the assignment requires.
    plan: Plan | None = None


class ExecuteResponse(BaseModel):
    post: dict | None
    image: dict | None
    trace: dict


class ReviseRequest(BaseModel):
    """Re-run ONLY the Editor Agent over a post that already exists.

    Deliberately does not take a plan: re-running the whole DAG to improve
    wording would re-issue both searches and regenerate the image, which costs
    ~10s and a Cloudflare neuron budget for a change that only touches text.
    """

    topic: str
    post: dict  # a previous response's `post` -- headline / body / hashtags


class ReviseResponse(BaseModel):
    post: dict | None
    ms: int
