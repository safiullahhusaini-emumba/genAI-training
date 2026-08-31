"""Query -> small preferences JSON.

One LLM call, with a deterministic rule-based fallback so the app still works
when Groq is rate-limited or the model returns garbage. The fallback is not
decoration: it is what runs in the failure case documented in the README.
"""

from __future__ import annotations

import difflib
import re

from config import MODEL_FAST
from src.llm import chat_json
from src.sources import CATEGORIES, CITIES, PRICE_LEVELS

SYSTEM = """You extract travel preferences from a user's question.

Return exactly this shape:
{
  "city": "<one city name, or null if not stated>",
  "categories": ["food" | "art" | "sightseeing", ...],
  "price_level": "cheap" | "medium" | "expensive" | null,
  "duration_days": <integer or null>,
  "keywords": ["<up to 5 concrete nouns from the query>"]
}

Rules:
- Never invent a city. If the user did not name one, use null.
- "budget", "cheap eats", "on a shoestring" -> cheap. "splurge", "fine dining" -> expensive.
- Only use the three category values listed. Museums and galleries are "art".
  Landmarks, parks and walking are "sightseeing". Anything eaten or drunk is "food".
- categories may be empty if the query is generic."""

CHEAP_WORDS = ["cheap", "budget", "affordable", "shoestring", "free", "inexpensive", "low cost"]
PRICEY_WORDS = ["expensive", "luxury", "splurge", "fine dining", "upscale", "high end", "michelin"]
CATEGORY_WORDS = {
    "food": ["food", "eat", "restaurant", "cafe", "dining", "cuisine", "beer", "bar", "coffee"],
    "art": ["art", "museum", "gallery", "exhibition", "street art", "sculpture"],
    "sightseeing": ["see", "sight", "landmark", "walk", "park", "tour", "monument", "history"],
}
_DAYS = re.compile(r"(\d+)\s*[- ]?\s*day")


def _word_in(word: str, low: str) -> bool:
    """Word-boundary check: plain `in` lets "art" match inside "start". The
    trailing `s?` keeps simple plurals ("eat" -> "eats") matching too."""
    return re.search(r"(?<!\w)" + re.escape(word) + r"s?(?!\w)", low) is not None


def _rules(query: str) -> dict:
    low = query.lower()
    # Word boundaries, not substrings: "medinas of Marrakesh" should not
    # resolve to the city of Medina.
    city = next((c for c in CITIES if re.search(rf"(?<!\w){re.escape(c.lower())}(?!\w)", low)), None)
    cats = [c for c, words in CATEGORY_WORDS.items() if any(_word_in(w, low) for w in words)]
    price = None
    if any(_word_in(w, low) for w in CHEAP_WORDS):
        price = "cheap"
    elif any(_word_in(w, low) for w in PRICEY_WORDS):
        price = "expensive"
    days = _DAYS.search(low)
    return {
        "city": city,
        "categories": cats,
        "price_level": price,
        "duration_days": int(days.group(1)) if days else None,
        "keywords": [w for w in re.findall(r"[a-z]{4,}", low)][:5],
        "_source": "rules",
    }


# A one-letter slip in a city name is indistinguishable from an unindexed city
# once it reaches the payload filter — both match zero points, and the city
# filter is never relaxed. "pehawar" therefore refused a question the corpus
# could answer. Snap near-misses onto the real name; leave genuine misses alone
# so an unindexed city still refuses deterministically.
#
# Calibrated against the corpus: real typos score 0.87-0.94 ("pehawar"/Peshawar
# 0.93, "berln"/Berlin 0.91, "istanbal"/Istanbul 0.88), while the closest an
# absent city gets is 0.55 ("delhi"/Berlin). Anything in between is a coin flip
# we would rather lose loudly than win silently, hence the note below.
_CITY_FUZZ = 0.85


def _snap_city(city: str) -> tuple[str, str | None]:
    """Returns (city, note). Note is set only when the name was corrected."""
    low = city.strip().lower()
    exact = next((c for c in CITIES if c.lower() == low), None)
    if exact:
        return exact, None
    near = difflib.get_close_matches(low, [c.lower() for c in CITIES], n=1, cutoff=_CITY_FUZZ)
    if near:
        corrected = next(c for c in CITIES if c.lower() == near[0])
        return corrected, f"Read '{city.strip()}' as {corrected}."
    return city.strip().title(), None


def _validate(prefs: dict) -> dict:
    city, city_note = prefs.get("city"), None
    if isinstance(city, str) and city.strip():
        city, city_note = _snap_city(city)
    else:
        city = None

    cats = [c for c in (prefs.get("categories") or []) if c in CATEGORIES]
    price = prefs.get("price_level")
    price = price if price in PRICE_LEVELS else None

    days = prefs.get("duration_days")
    days = days if isinstance(days, int) and 0 < days < 60 else None

    kws = [str(k) for k in (prefs.get("keywords") or [])][:5]

    out = {
        "city": city,
        "categories": cats,
        "price_level": price,
        "duration_days": days,
        "keywords": kws,
    }
    if city_note:
        out["_note"] = city_note
    return out


def extract(query: str, model: str = MODEL_FAST) -> dict:
    try:
        raw = chat_json(SYSTEM, f"Query: {query}", model=model, temperature=0.0, max_tokens=500)
    except Exception as exc:  # network, auth, rate limit
        out = _rules(query)
        out["_note"] = f"LLM extraction failed ({type(exc).__name__}); used rules."
        return out

    if not raw:
        out = _rules(query)
        out["_note"] = "LLM returned unparseable JSON; used rules."
        return out

    prefs = _validate(raw)
    prefs["_source"] = "llm"

    # Belt and braces: the 8B model drops the city about 1 in 15 times.
    if not prefs["city"]:
        fallback_city = _rules(query)["city"]
        if fallback_city:
            prefs["city"] = fallback_city
            prefs["_note"] = "City recovered by rule fallback."
    return prefs
