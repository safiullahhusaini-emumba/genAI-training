"""Streamlit UI.

The layout is built around the one thing the assignment asks the UI to prove:
that query -> retrieval -> reasoning -> answer is legible. So the pipeline strip
sits between the question and the answer, always visible, with real timings —
not hidden behind an expander.
"""

from __future__ import annotations

from html import escape

import streamlit as st
from markdown_it import MarkdownIt

from config import BUILD_CMD, QDRANT_COLLECTION, QDRANT_DASHBOARD_URL, QDRANT_URL
from src import judge
from src.pipeline import get_store, run
from src.sources import SOURCES
from src.store import CollectionMissing, StoreUnavailable

DOCKER_FIX = (
    f"**Qdrant is not reachable at `{QDRANT_URL}`.**\n\n"
    "The vector database runs as a container now. Start it, then reload:\n\n"
    "```bash\ndocker compose up -d\ndocker compose ps      # wait for (healthy)\n```"
)
BUILD_FIX = (
    f"**Qdrant is running, but the `{QDRANT_COLLECTION}` collection is empty or missing.**\n\n"
    "Different problem, different fix — the server is fine, the data isn't there:\n\n"
    f"```bash\n{BUILD_CMD}\n```"
)

# The answer card is a styled <div>, so the answer's markdown has to be real
# HTML by the time it gets there: splice raw markdown into an HTML block and
# CommonMark treats everything up to the first blank line as literal HTML —
# which is why the first day of an itinerary used to collapse into one run-on
# paragraph while later days rendered as proper lists. html=False also means a
# model that emits a stray tag gets it escaped rather than injected.
_MD = (
    MarkdownIt("commonmark", {"html": False, "linkify": True})
    .enable("table")
    .enable("strikethrough")
)

st.set_page_config(page_title="Travel RAG", page_icon="◈", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Archivo:wght@400;500;600;700&display=swap');

:root {
  --ink:      #0E1A22;
  --ink-soft: #4A5C68;
  --rule:     #C9D4DB;
  --paper:    #EDF1F4;
  --card:     #FFFFFF;
  --signal:   #0B6E6E;
  --warn:     #9A6410;
}

html, body, [class*="css"] { font-family: 'Archivo', system-ui, sans-serif; }
.stApp { background: var(--paper); }

.masthead { border-bottom: 2px solid var(--ink); padding-bottom: .6rem; margin-bottom: 1.4rem; }
.masthead h1 {
  font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.9rem;
  letter-spacing: -.02em; color: var(--ink); margin: 0;
}
.masthead p {
  font-family: 'IBM Plex Mono', monospace; font-size: .74rem; letter-spacing: .09em;
  text-transform: uppercase; color: var(--ink-soft); margin: .3rem 0 0;
}

/* the signature element: a departure-board style strip of pipeline stages */
.strip { display: flex; gap: 0; margin: 1.1rem 0 1.6rem; flex-wrap: wrap; }
.stage {
  flex: 1 1 130px; background: var(--card); border: 1px solid var(--rule);
  border-right: none; padding: .6rem .8rem;
}
.stage:last-child { border-right: 1px solid var(--rule); }
.stage .name {
  font-family: 'IBM Plex Mono', monospace; font-size: .66rem; letter-spacing: .1em;
  text-transform: uppercase; color: var(--ink-soft);
}
.stage .val {
  font-family: 'IBM Plex Mono', monospace; font-size: .88rem; font-weight: 500;
  color: var(--ink); margin-top: .22rem;
}
.stage .ms {
  font-family: 'IBM Plex Mono', monospace; font-size: .66rem;
  color: var(--signal); margin-top: .18rem;
}
.stage.flag { border-top: 3px solid var(--warn); }
.stage.flag .ms { color: var(--warn); }

.answer {
  background: var(--card); border: 1px solid var(--rule);
  border-left: 3px solid var(--signal); padding: 1.3rem 1.5rem; line-height: 1.62;
}
.answer > :first-child { margin-top: 0; }
.answer > :last-child  { margin-bottom: 0; }
.answer h1, .answer h2, .answer h3, .answer h4 {
  font-size: 1.02rem; font-weight: 700; color: var(--ink);
  margin: 1.35rem 0 .5rem; letter-spacing: -.01em;
}
.answer p  { margin: 0 0 .7rem; }
.answer ul, .answer ol { margin: 0 0 .8rem; padding-left: 1.35rem; }
.answer li { margin-bottom: .3rem; }
.answer code {
  font-family: 'IBM Plex Mono', monospace; font-size: .86em;
  background: var(--paper); padding: .05rem .3rem;
}

.chunk {
  background: var(--card); border: 1px solid var(--rule);
  padding: .8rem .95rem; margin-bottom: .6rem;
}
.chunk .head {
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .05em;
  color: var(--ink-soft); margin-bottom: .4rem;
}
.chunk .score { color: var(--signal); font-weight: 600; }
.chunk .body { font-size: .84rem; line-height: 1.5; color: var(--ink); }

.prefkey {
  font-family: 'IBM Plex Mono', monospace; font-size: .74rem;
  color: var(--ink-soft); letter-spacing: .04em;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="masthead">
  <h1>Preference-Aware Travel RAG</h1>
  <p>query → preferences → hybrid retrieval → rerank → judge → grounded answer</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Connecting to Qdrant…")
def _store():
    return get_store()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("**Retrieval settings**")
    use_hyde = st.toggle(
        "HyDE query expansion",
        value=True,
        help="Write a hypothetical guide paragraph and search with that too. "
        "Helps vague queries, adds one fast LLM call.",
    )

    st.divider()
    st.markdown("**Index**")
    # Fails soft on purpose: the sidebar renders before the user has typed
    # anything, so "Qdrant is down" should be visible up front rather than
    # ambushing them after they press Plan it.
    try:
        st.caption(f"`{QDRANT_COLLECTION}` · {_store().size:,} chunks")
        st.caption(f"[qdrant dashboard]({QDRANT_DASHBOARD_URL}/dashboard)")
    except StoreUnavailable:
        st.error("Qdrant unreachable — run `docker compose up -d`")
    except CollectionMissing:
        st.warning(f"Collection empty — run `{BUILD_CMD}`")

    st.divider()
    st.markdown("**Indexed sources**")
    for src in SOURCES:
        st.caption(f"{src['city']} · {src['url'].rsplit('/', 1)[-1].replace('_', ' ')}")

# ---------------------------------------------------------------- input
EXAMPLES = [
    "3-day Berlin trip with cheap food and art",
    "Where should I eat in Lisbon on a tight budget?",
    "Rainy afternoon in Amsterdam, into modern art",
    "Best ramen in Tokyo",  # not in the corpus at all — the city filter refuses deterministically
]

# A form, so pressing Enter in the box submits. With a bare text_input the
# rerun that Enter triggers leaves the button False, and the app looks dead.
with st.form("ask", clear_on_submit=False, border=False):
    col_a, col_b = st.columns([3, 1])
    with col_a:
        query = st.text_input(
            "Your question",
            placeholder="3-day Berlin trip with cheap food and art",
            label_visibility="collapsed",
        )
    with col_b:
        go = st.form_submit_button("Plan it", use_container_width=True, type="primary")

st.caption("Try: " + "  ·  ".join(f"*{e}*" for e in EXAMPLES))

if go and query.strip():
    try:
        store = _store()
    except StoreUnavailable:
        st.error(DOCKER_FIX)
        st.stop()
    except CollectionMissing:
        st.error(BUILD_FIX)
        st.stop()

    try:
        with st.spinner("Retrieving and reasoning…"):
            st.session_state["result"] = run(query.strip(), store=store, use_hyde=use_hyde)
    except StoreUnavailable:
        # The cached client can outlive its server — Qdrant restarted, or the
        # laptop slept and the connection pool went stale. Drop the dead client
        # and retry once; the session_state guard stops a genuinely-down server
        # from looping this into an infinite rerun.
        if not st.session_state.get("_reconnected"):
            st.session_state["_reconnected"] = True
            st.cache_resource.clear()
            st.rerun()
        st.error(DOCKER_FIX)
        st.stop()
    except CollectionMissing:
        # e.g. someone ran `docker compose down -v` mid-session.
        st.error(BUILD_FIX)
        st.stop()

    # The retry budget is per-outage, not per-session: without this, a second
    # Qdrant restart later in the same session would never get its reconnect.
    st.session_state.pop("_reconnected", None)

# Streamlit re-executes this whole script on *any* rerun — flipping the HyDE
# toggle, a browser reconnect, or a source file changing under the dev bind
# mount. The submit button is False on all of those, so reading the result
# straight off the run above meant the answer blanked the moment you touched
# anything. Park it in session_state and re-render from there instead.
result = st.session_state.get("result")
if result is None:
    st.stop()

trace = result["trace"]

# ---------------------------------------------------------------- pipeline strip
cells = []
for stage in trace["stages"]:
    detail = str(stage["detail"] or "—")
    # Anything that made the pipeline work harder or give up gets the warn rule:
    # the retry itself, a failed context check, and a refusal.
    flag = (
        stage["stage"].startswith("retry")
        or detail in ("refused", judge.INSUFFICIENT)
    )
    cells.append(
        f'<div class="stage{" flag" if flag else ""}">'
        f'<div class="name">{escape(stage["stage"])}</div>'
        f'<div class="val">{escape(detail)}</div>'
        f'<div class="ms">{stage["ms"]} ms</div></div>'
    )
cells.append(
    f'<div class="stage"><div class="name">total</div>'
    f'<div class="val">{escape(str(trace["outcome"]))}</div>'
    f'<div class="ms">{trace["total_ms"]} ms</div></div>'
)
st.markdown(f'<div class="strip">{"".join(cells)}</div>', unsafe_allow_html=True)

if trace["relaxed_retry"]:
    st.warning(
        "First pass failed the context check, so filters were relaxed once "
        f"({trace['judge'].get('reason', '')})"
    )

# ---------------------------------------------------------------- answer
st.markdown(
    f'<div class="answer">{_MD.render(result["answer"] or "_The model returned nothing._")}</div>',
    unsafe_allow_html=True,
)

if result["sources"]:
    st.markdown("**Sources**")
    # FINAL_K chunks often come from the same page, so list each URL once and
    # show every citation number that points at it — five identical links under
    # five different numbers reads like five sources.
    by_url: dict[str, list[int]] = {}
    for src in result["sources"]:
        by_url.setdefault(src["url"], []).append(src["n"])
    for url, nums in by_url.items():
        marks = "".join(f"`[{n}]`" for n in nums)
        st.markdown(f"{marks} [{url}]({url})")

# ---------------------------------------------------------------- debug
st.divider()
with st.expander("Debug panel", expanded=True):
    tab_prefs, tab_chunks, tab_urls, tab_raw = st.tabs(
        ["Preferences", "Retrieved chunks", "URLs used", "Raw trace"]
    )

    with tab_prefs:
        prefs = trace["preferences"]
        cols = st.columns(4)
        cols[0].metric("City", prefs.get("city") or "—")
        cols[1].metric("Budget", prefs.get("price_level") or "any")
        cols[2].metric("Days", prefs.get("duration_days") or "—")
        cols[3].metric("Categories", ", ".join(prefs.get("categories") or []) or "any")
        st.markdown(
            f'<div class="prefkey">extracted by: {escape(str(prefs.get("_source", "?")))}'
            + (f' — {escape(str(prefs["_note"]))}' if prefs.get("_note") else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.json(prefs)

    with tab_chunks:
        r = trace.get("retrieval_relaxed") or trace["retrieval"]
        st.caption(
            f"filter {r.get('filter_summary') or 'none'} → "
            f"dense {r['dense_hits']} + bm25 {r['bm25_hits']} "
            f"→ server-side RRF {r['fused_pool']} → top {len(trace['chunks'])}"
            + ("  ·  HyDE on" if r.get("hyde_used") else "  ·  HyDE off")
        )
        for i, chunk in enumerate(trace["chunks"], start=1):
            # Corpus text is scraped HTML pages; an unescaped "<" or "&" in a
            # chunk silently eats the rest of the card.
            st.markdown(
                f'<div class="chunk"><div class="head">'
                f'[{i}] {escape(str(chunk["city"]))} · {escape(str(chunk["category"]))} · '
                f'{escape(str(chunk["price_level"]))} · '
                f'<span class="score">rerank {escape(str(chunk.get("rerank_score", "–")))}</span> · '
                f'fused {chunk["retrieval_score"]} · {escape(str(chunk.get("rerank_reason", "")))}'
                f'</div><div class="body">{escape(chunk["text"][:700])}…</div></div>',
                unsafe_allow_html=True,
            )
        if r.get("hyde_doc"):
            st.markdown("**HyDE probe document (never shown to the user)**")
            st.text(r["hyde_doc"])

    with tab_urls:
        for url in trace["urls"]:
            st.markdown(f"- [{url}]({url})")

    with tab_raw:
        st.json({k: v for k, v in trace.items() if k != "chunks"})
