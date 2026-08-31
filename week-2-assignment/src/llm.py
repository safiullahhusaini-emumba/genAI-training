"""Thin Groq wrapper. Every LLM call in this project goes through here."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from groq import Groq

from config import GROQ_API_KEY, MODEL_FAST

_FENCE = re.compile(r"^```(?:json)?|```$", re.M)


@lru_cache(maxsize=1)
def client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env.")
    return Groq(api_key=GROQ_API_KEY)


def _reasoning(model: str, effort: str) -> dict:
    """gpt-oss models on Groq are reasoning models, and their reasoning tokens
    count against max_tokens whether or not they are returned. Left at Groq's
    default they routinely burn 250+ tokens before the answer starts, which
    truncates short-budget calls mid-sentence (finish_reason='length'). Capping
    the effort and hiding the trace keeps the budget for the visible output.

    The installed groq SDK predates typed reasoning_* params, but Groq's REST
    API accepts them; extra_body passes them through untyped.
    """
    if "gpt-oss" not in model:
        return {}
    return {"extra_body": {"reasoning_effort": effort, "reasoning_format": "hidden"}}


def chat(
    system: str,
    user: str,
    model: str = MODEL_FAST,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    reasoning_effort: str = "low",
    **extra,
) -> str:
    resp = client().chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **_reasoning(model, reasoning_effort),
        **extra,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_json(system: str, user: str, model: str = MODEL_FAST, **kw) -> dict:
    """Ask for JSON, survive the usual failure modes (fences, preamble, trailing prose).

    The reasoning suppression that keeps the JSON from being truncated now lives
    in chat(), since every caller needs it, not just this one.
    """
    raw = chat(system + "\n\nRespond with JSON only. No prose, no code fences.",
               user, model=model, **kw)
    cleaned = _FENCE.sub("", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}
