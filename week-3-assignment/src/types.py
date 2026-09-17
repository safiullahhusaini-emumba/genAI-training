"""Shared types between tools and the executor. Split out to avoid a circular
import: tools need the result shape to read their dependencies' outputs, and
the executor needs it to build the trace -- neither should import the other.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepResult:
    """What one executed plan step produced. This is the unit the trace is
    built from and the shape every tool's dependencies arrive in."""

    step: int
    tool: str
    ok: bool
    output: dict | None = None
    error: str | None = None
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def ms(self) -> int:
        return int((self.ended_at - self.started_at) * 1000)

    def to_trace(self) -> dict:
        return {
            "step": self.step,
            "tool": self.tool,
            "ok": self.ok,
            "ms": self.ms,
            "error": self.error,
        }
