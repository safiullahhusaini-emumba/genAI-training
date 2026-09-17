"""The tool registry. The planner prompt (src/planner.py) and the validator
(src/validator.py) both build their "known tools" list from this dict rather
than hand-maintaining a parallel list, so the planner can never be told about
a tool the executor doesn't actually have -- and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from pydantic import BaseModel

from src.tools import editor, generator, image_gen, social_search, summarizer, web_search
from src.types import StepResult

ToolFn = Callable[[str, dict, dict[int, StepResult]], Awaitable[dict]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_schema: type[BaseModel]
    fn: ToolFn
    # True for tools whose entire job is to transform what their dependencies
    # produced, so they are meaningless with an empty depends_on -- a
    # summarizer with nothing to summarize is not a step, it's a no-op that
    # then looks like a *failed* dependency to whatever consumes it. The
    # validator prunes these; see src/validator.py. Source tools (searches,
    # image generation) and the terminal agents are all fine with no deps.
    requires_deps: bool = False


TOOLS: dict[str, Tool] = {
    web_search.NAME: Tool(web_search.NAME, web_search.DESCRIPTION, web_search.WebSearchArgs, web_search.run),
    social_search.NAME: Tool(
        social_search.NAME, social_search.DESCRIPTION, social_search.SocialSearchArgs, social_search.run
    ),
    summarizer.NAME: Tool(
        summarizer.NAME,
        summarizer.DESCRIPTION,
        summarizer.SummarizerArgs,
        summarizer.run,
        requires_deps=True,
    ),
    generator.NAME: Tool(generator.NAME, generator.DESCRIPTION, generator.GeneratorArgs, generator.run),
    editor.NAME: Tool(editor.NAME, editor.DESCRIPTION, editor.EditorArgs, editor.run),
    image_gen.NAME: Tool(image_gen.NAME, image_gen.DESCRIPTION, image_gen.ImageGenArgs, image_gen.run),
}


def tool_names() -> list[str]:
    return list(TOOLS.keys())


def describe_tools() -> str:
    """Rendered for the planner's system prompt: name, description, and the
    argument schema each tool accepts, generated from the registry so it
    can never drift out of sync with what the executor can actually call."""
    lines = []
    for tool in TOOLS.values():
        schema = tool.args_schema.model_json_schema()
        props = schema.get("properties", {})
        arg_bits = ", ".join(f"{k}: {v.get('description', v.get('type', ''))}" for k, v in props.items())
        lines.append(f"- {tool.name}: {tool.description}" + (f" Args: {{{arg_bits}}}" if arg_bits else ""))
    return "\n".join(lines)
