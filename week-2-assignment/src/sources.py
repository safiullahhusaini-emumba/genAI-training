"""The fixed seed list.

Metadata is attached at the *source* level and inherited by every chunk from
that page. `category` and `price_level` get refined per-chunk in ingest.py.
"""

from __future__ import annotations

SOURCES: list[dict] = [
    {
        "url": "https://en.wikivoyage.org/wiki/Berlin",
        "city": "Berlin",
        "category": "sightseeing",
        "price_level": "medium",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Berlin/Mitte",
        "city": "Berlin",
        "category": "sightseeing",
        "price_level": "medium",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Berlin/Kreuzberg",
        "city": "Berlin",
        "category": "food",
        "price_level": "cheap",
    },
    {
        # Wikivoyage merged the standalone Friedrichshain article into this
        # combined district page; the old /Berlin/Friedrichshain URL 404s.
        "url": "https://en.wikivoyage.org/wiki/Berlin/Friedrichshain-Kreuzberg",
        "city": "Berlin",
        "category": "food",
        "price_level": "cheap",
    },
    {
        # Wikivoyage folded Prenzlauer Berg into this larger district page;
        # the old /Berlin/Prenzlauer_Berg URL 404s.
        "url": "https://en.wikivoyage.org/wiki/Berlin/East",
        "city": "Berlin",
        "category": "art",
        "price_level": "medium",
    },
    # Museum_Island dropped: Wikivoyage folded it into Berlin/Mitte (already
    # indexed above), and the standalone /wiki/Museum_Island URL 404s.
    {
        "url": "https://en.wikivoyage.org/wiki/Amsterdam",
        "city": "Amsterdam",
        "category": "sightseeing",
        "price_level": "medium",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Prague",
        "city": "Prague",
        "category": "sightseeing",
        "price_level": "cheap",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Lisbon",
        "city": "Lisbon",
        "category": "food",
        "price_level": "cheap",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Istanbul",
        "city": "Istanbul",
        "category": "food",
        "price_level": "cheap",
    },
    # Pakistan
    {
        "url": "https://en.wikivoyage.org/wiki/Lahore",
        "city": "Lahore",
        "category": "sightseeing",
        "price_level": "cheap",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Karachi",
        "city": "Karachi",
        "category": "food",
        "price_level": "cheap",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Islamabad",
        "city": "Islamabad",
        "category": "sightseeing",
        "price_level": "medium",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Peshawar",
        "city": "Peshawar",
        "category": "sightseeing",
        "price_level": "cheap",
    },
    # Saudi Arabia
    {
        "url": "https://en.wikivoyage.org/wiki/Riyadh",
        "city": "Riyadh",
        "category": "sightseeing",
        "price_level": "expensive",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Jeddah",
        "city": "Jeddah",
        "category": "food",
        "price_level": "medium",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Mecca",
        "city": "Mecca",
        "category": "sightseeing",
        "price_level": "medium",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Medina",
        "city": "Medina",
        "category": "sightseeing",
        "price_level": "medium",
    },
    # Other major Islamic-world cities
    {
        "url": "https://en.wikivoyage.org/wiki/Cairo",
        "city": "Cairo",
        "category": "sightseeing",
        "price_level": "cheap",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Dubai",
        "city": "Dubai",
        "category": "sightseeing",
        "price_level": "expensive",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Doha",
        "city": "Doha",
        "category": "sightseeing",
        "price_level": "expensive",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Amman",
        "city": "Amman",
        "category": "sightseeing",
        "price_level": "medium",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Kuala_Lumpur",
        "city": "Kuala Lumpur",
        "category": "food",
        "price_level": "cheap",
    },
    {
        "url": "https://en.wikivoyage.org/wiki/Marrakesh",
        "city": "Marrakesh",
        "category": "sightseeing",
        "price_level": "cheap",
    },
]

CITIES = sorted({s["city"] for s in SOURCES})
CATEGORIES = ["food", "art", "sightseeing"]
PRICE_LEVELS = ["cheap", "medium", "expensive"]
