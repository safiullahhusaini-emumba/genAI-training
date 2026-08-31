"""Final answer. Grounded, cited, and honest about gaps."""

from __future__ import annotations

import re

from config import MODEL_SMART
from src.llm import chat
from src.sources import CITIES

SYSTEM = """You are a travel planner who only uses the sources provided.

Rules:
- Every venue, price, opening time or neighbourhood you name must appear in the sources.
- Cite with an ASCII bracketed source number, like [2], right after the claim it
  supports. Plain [ and ] only — never fullwidth or CJK brackets.
- If the sources don't cover part of the question, say so in one line instead of filling it in.
- Match the user's stated budget and interests. If they said cheap, do not suggest a splurge.
- Structure a multi-day request by day, one markdown `###` heading per day.
- Never place the same venue on two different days, and never claim two names
  refer to different places unless a source says so.
- Nothing in parentheses unless the source says it: no invented "part of X"
  or "also known as Y" asides.
- Keep it tight — no filler, no "as an AI".
- Write in the second person, plainly. No emoji."""

# The refusal has to know what the corpus *does* hold, or it suggests another
# question about the same unindexed city — which fails identically.
REFUSAL_SYSTEM = f"""You explain, in three sentences maximum, that you can't answer a travel
question because the indexed sources don't cover it. Name what is missing, then suggest
one question the user could ask instead. Do not answer from general knowledge.

The index covers only these cities: {", ".join(CITIES)}. Your suggested question must
be about one of them — never about a city absent from that list."""


def _format_sources(chunks: list[dict]) -> tuple[str, list[dict]]:
    blocks, used = [], []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] source: {chunk['url']} | city: {chunk['city']} | "
            f"category: {chunk['category']} | price: {chunk['price_level']}\n{chunk['text']}"
        )
        used.append({"n": i, "url": chunk["url"], "city": chunk["city"]})
    return "\n\n---\n\n".join(blocks), used


# The prompt asks for ASCII [2], but gpt-oss reaches for fullwidth brackets often
# enough to matter — and a citation the UI's source list can't be matched against
# is a broken citation. Cheaper to normalise than to re-prompt.
_ALT_CITE = re.compile(r"[\u3010\u3014\uff3b]\s*(\d{1,2})\s*[\u3011\u3015\uff3d]")


def _normalise_citations(text: str) -> str:
    return _ALT_CITE.sub(r"[\1]", text)


def answer(query: str, chunks: list[dict], prefs: dict, model: str = MODEL_SMART) -> dict:
    sources, used = _format_sources(chunks)
    pref_line = ", ".join(
        f"{k}={v}" for k, v in prefs.items()
        if v and not k.startswith("_") and k != "keywords"
    ) or "none extracted"

    user = (
        f"Question: {query}\n\n"
        f"Extracted preferences: {pref_line}\n\n"
        f"Sources:\n\n{sources}"
    )
    text = chat(SYSTEM, user, model=model, temperature=0.3, max_tokens=1400)
    return {"answer": _normalise_citations(text), "sources": used}


def refuse(query: str, reason: str, model: str = MODEL_SMART) -> dict:
    text = chat(
        REFUSAL_SYSTEM,
        f"Question: {query}\nWhy the retrieval failed: {reason}",
        model=model,
        temperature=0.2,
        max_tokens=500,   # three sentences is ~120; the rest is slack against a long reasoning pass
    )
    return {"answer": text, "sources": []}
