"""The plan artifact: what the Planner Agent produces and the executor consumes.

The planner's output schema lives here on its own -- kept in one small
file with no LLM code in it, so the shape of a plan is auditable independent of
how one gets produced.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Step(BaseModel):
    step: int = Field(..., ge=1, description="1-indexed step id, unique within the plan")
    tool: str = Field(..., description="name of a tool in src.tools.TOOLS")
    depends_on: list[int] = Field(default_factory=list, description="step ids that must complete first")
    args: dict = Field(default_factory=dict, description="tool-specific arguments, e.g. time_range")

    @field_validator("depends_on")
    @classmethod
    def _no_self_dependency(cls, v: list[int], info) -> list[int]:
        step_id = info.data.get("step")
        if step_id is not None and step_id in v:
            raise ValueError(f"step {step_id} cannot depend on itself")
        return v


class Plan(BaseModel):
    steps: list[Step]


# The deterministic fallback, following this project's rule that every LLM
# decision has a non-LLM answer it can fall back to. Used when the planner
# returns garbage the validator can't repair -- see src/validator.py. It is the
# most generally useful shape: search both sources, condense, write, edit,
# with the image on its own parallel branch.
DEFAULT_PLAN = Plan(
    steps=[
        Step(step=1, tool="web_search", depends_on=[], args={}),
        Step(step=2, tool="social_search", depends_on=[], args={}),
        Step(step=3, tool="summarizer", depends_on=[1, 2], args={}),
        Step(step=4, tool="content_generator", depends_on=[3], args={}),
        Step(step=5, tool="content_editor", depends_on=[4], args={}),
        Step(step=6, tool="image_generator", depends_on=[], args={}),
    ]
)

# The names below must exist in src.tools.TOOLS. Kept here (rather than only
# in the registry) so plan_schema.py stays importable without pulling in every
# tool's network client -- validator.py imports both and cross-checks them.
GENERATOR_TOOL = "content_generator"
EDITOR_TOOL = "content_editor"
