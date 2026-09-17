"""The one place every LLM call in this project goes through.

Wraps langchain_groq.ChatGroq -- the assignment requires LangChain, and this is
the one file where that requirement actually touches the code; everything
downstream of here (planner, tools, generator, editor) just calls chat()/chat_json().
Keeping it to a single choke point means retry/parsing/fallback behaviour is
defined once instead of drifting per caller.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from config import GROQ_API_KEY, MODEL_FAST, REASONING_EFFORT, REASONING_FORMAT

_FENCE = re.compile(r"^```(?:json)?|```$", re.M)

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=8)
def _client(model: str, temperature: float, max_tokens: int) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env.")
    kwargs = dict(
        model=model,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # gpt-oss models only: reasoning_effort/format are rejected by non-reasoning
    # Groq models (e.g. llama-3.3-*), so only pass them when the model wants them.
    # langchain-groq 1.1.3 exposes these as native ChatGroq fields, so they can
    # be passed directly rather than smuggled through an extra_body passthrough.
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = REASONING_EFFORT
        kwargs["reasoning_format"] = REASONING_FORMAT
    return ChatGroq(**kwargs)


def chat(
    system: str,
    user: str,
    model: str = MODEL_FAST,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Plain text completion. Returns "" on any failure rather than raising --
    every caller in this project has a deterministic fallback for empty output."""
    try:
        llm = _client(model, temperature, max_tokens)
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return (resp.content or "").strip()
    except Exception:
        return ""


def chat_json(
    system: str,
    user: str,
    schema: type[T],
    model: str = MODEL_FAST,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> T | None:
    """Ask for JSON matching `schema`, survive the usual failure modes.

    Two-tier strategy (the second tier's existence was established by
    testing, not planning -- see the note below):
      1. LangChain's with_structured_output(method="json_schema") -- Groq's
         own Structured Output API, which enforces the schema server-side.
         Verified working for both gpt-oss-20b and gpt-oss-120b during
         development. This is NOT the path broken by langchain#34155 (that's
         the default method="function_calling"/tool-calling strategy, which
         this deliberately avoids) -- json_schema is a separate code path.
      2. Manual: ask for raw JSON with the schema spelled out in the prompt,
         then strip the failure shapes Groq actually emits (code fences, a
         prose preamble, trailing commentary) before validating.

    A real bug lived here during development: the original tier 1 used
    method="json_mode", which -- per LangChain's own docstring -- tells Groq
    "emit valid JSON" but never tells the model what shape to emit. It
    produced perfectly valid JSON with invented field names (e.g. {"post":
    "..."} instead of {"headline", "body", "hashtags"}), which failed Pydantic
    validation on every call. method="json_schema" fixes tier 1 outright, and
    tier 2 now also renders the schema into the prompt for the same reason.

    Returns None -- never raises -- if both tiers fail, so the caller (planner,
    validator, generator, editor, art director) can fall back to its own
    deterministic default. That fallback is what makes this safe to call from
    inside the executor's per-step-never-raises contract.
    """
    try:
        llm = _client(model, temperature, max_tokens)
        structured = llm.with_structured_output(schema, method="json_schema")
        result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        if isinstance(result, schema):
            return result
    except Exception:
        pass  # fall through to the manual parse below

    schema_hint = _schema_hint(schema)
    raw = chat(
        f"{system}\n\n{schema_hint}\n\nRespond with JSON only. No prose, no code fences.",
        user,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not raw:
        return None
    cleaned = _FENCE.sub("", raw).strip()
    data = _loads_lenient(cleaned)
    if data is None:
        return None
    try:
        return schema.model_validate(data)
    except Exception:
        return None


def _schema_hint(schema: type[BaseModel]) -> str:
    """Render a Pydantic model's fields into a plain-language instruction.
    Needed for the tier-2 manual-JSON fallback, which (unlike json_schema
    mode) has no other way to tell the model what shape to emit."""
    props = schema.model_json_schema().get("properties", {})
    fields = ", ".join(f'"{k}": {v.get("description", v.get("type", "any"))}' for k, v in props.items())
    return f"Respond with a JSON object with exactly these fields: {{{fields}}}"


def _loads_lenient(cleaned: str) -> dict | None:
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
