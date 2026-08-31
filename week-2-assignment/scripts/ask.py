"""Run one query through the pipeline from the terminal — handy for iterating on
prompts without a Streamlit reload in the way.

    python scripts/ask.py "3-day Berlin trip with cheap food and art"
    python scripts/ask.py "cheap eats in Lisbon" --no-hyde
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--no-hyde", action="store_true")
    ap.add_argument("--trace", action="store_true", help="dump the full trace as JSON")
    args = ap.parse_args()

    result = run(args.query, use_hyde=not args.no_hyde)
    trace = result["trace"]

    print("\n" + "=" * 72)
    for stage in trace["stages"]:
        # detail can be None (e.g. preferences with no _source); :<30 on None raises.
        print(f"  {stage['stage']:<26} {str(stage['detail'] or '-'):<30} {stage['ms']:>5} ms")
    print(f"  {'TOTAL':<26} {trace['outcome']:<30} {trace['total_ms']:>5} ms")
    print("=" * 72)

    print(f"\nPreferences: {json.dumps(trace['preferences'], ensure_ascii=False)}")
    print(f"URLs used:   {len(trace['urls'])}")
    for url in trace["urls"]:
        print(f"  - {url}")

    print("\n" + result["answer"] + "\n")

    if args.trace:
        print(json.dumps({k: v for k, v in trace.items() if k != "chunks"}, indent=2))


if __name__ == "__main__":
    main()
