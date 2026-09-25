"""Streamlit UI -- a drafting client only; it does not post to LinkedIn.

Flow: topic -> POST /plan -> render the DAG as an editable checklist -> POST
/execute/stream -> tools tick live as they finish -> final post, image, and a
debug panel. The two-call shape means the plan is visible, and editable,
before anything expensive runs -- which is also the single clearest way to
show that the planner's decision is real and not decorative.
"""

from __future__ import annotations

import base64
import json
import sys
from html import escape
from pathlib import Path

# `streamlit run ui/app.py` puts the SCRIPT's directory (ui/) on sys.path --
# see streamlit.web.bootstrap: `sys.path.insert(0, os.path.dirname(main_script_path))`
# -- and, because streamlit is an installed console script rather than
# `python app.py`, the working directory is NOT added. So the project root has
# to go on the path explicitly or `import config` fails with ModuleNotFoundError
# the moment a browser session connects (the HTTP health endpoint still answers
# fine, which makes this fail in a genuinely misleading way).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402  -- must follow the sys.path fix above
import streamlit as st  # noqa: E402

from config import API_BASE_URL, API_FIX_CMD  # noqa: E402

# Fail soft, and print the exact command that fixes it -- which differs
# depending on whether you're running under compose or bare.
API_DOWN_FIX = (
    f"**The API is not reachable at `{API_BASE_URL}`.**\n\n"
    "The backend runs as its own service. Start it, then try again:\n\n"
    f"```bash\n{API_FIX_CMD}\n```"
)

st.set_page_config(page_title="LinkedIn Content Curator", page_icon="◈", layout="wide")

CORE_TOOLS = {"content_generator", "content_editor"}

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
  --fail:     #B3261E;
}

html, body, [class*="css"] { font-family: 'Archivo', system-ui, sans-serif; }
.stApp { background: var(--paper); }

.masthead { border-bottom: 2px solid var(--ink); padding-bottom: .6rem; margin-bottom: 1.2rem; }
.masthead h1 {
  font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.8rem;
  letter-spacing: -.02em; color: var(--ink); margin: 0;
}
.masthead p {
  font-family: 'IBM Plex Mono', monospace; font-size: .74rem; letter-spacing: .09em;
  text-transform: uppercase; color: var(--ink-soft); margin: .3rem 0 0;
}

/* the signature element: a departure-board style strip of pipeline stages */
.strip { display: flex; gap: 0; margin: 1rem 0 1.2rem; flex-wrap: wrap; }
.stage {
  flex: 1 1 140px; background: var(--card); border: 1px solid var(--rule);
  border-right: none; padding: .55rem .75rem;
}
.stage:last-child { border-right: 1px solid var(--rule); }
.stage .name {
  font-family: 'IBM Plex Mono', monospace; font-size: .64rem; letter-spacing: .08em;
  text-transform: uppercase; color: var(--ink-soft);
}
.stage .val {
  font-family: 'IBM Plex Mono', monospace; font-size: .82rem; font-weight: 500;
  color: var(--ink); margin-top: .2rem;
}
.stage .ms {
  font-family: 'IBM Plex Mono', monospace; font-size: .64rem;
  color: var(--signal); margin-top: .16rem;
}
.stage.pending { opacity: .55; }
.stage.running { border-top: 3px solid var(--warn); }
.stage.running .ms { color: var(--warn); }
.stage.failed { border-top: 3px solid var(--fail); }
.stage.failed .ms { color: var(--fail); }
.stage.wave-label { flex: 0 0 auto; background: transparent; border: none; padding: .55rem .4rem .55rem 0; }

.answer {
  background: var(--card); border: 1px solid var(--rule);
  border-left: 3px solid var(--signal); padding: 1.2rem 1.4rem; line-height: 1.6;
}
.answer h3 { margin: 0 0 .6rem; font-size: 1.15rem; color: var(--ink); }
.answer .body { white-space: pre-wrap; color: var(--ink); }
.answer .tags { margin-top: .8rem; font-family: 'IBM Plex Mono', monospace; font-size: .82rem; color: var(--signal); }

.issue-card {
  background: var(--card); border: 1px solid var(--rule); border-left: 3px solid var(--warn);
  padding: .7rem .9rem; margin-bottom: .5rem; font-size: .86rem; color: var(--ink);
}

.plan-row {
  display: flex; align-items: center; gap: .6rem; background: var(--card);
  border: 1px solid var(--rule); padding: .5rem .8rem; margin-bottom: .4rem;
  font-family: 'IBM Plex Mono', monospace; font-size: .82rem;
}
.draft-card {
  background: var(--card); border: 1px solid var(--rule); border-left: 3px solid var(--ink-soft);
  padding: .8rem .95rem; font-size: .84rem; line-height: 1.5; color: var(--ink);
}
.draft-card.after { border-left-color: var(--signal); }
.draft-card .body { white-space: pre-wrap; margin-top: .45rem; }
.draft-card .tags {
  margin-top: .5rem; font-family: 'IBM Plex Mono', monospace;
  font-size: .76rem; color: var(--signal);
}

.plan-row .tool { font-weight: 600; color: var(--ink); min-width: 160px; }
.plan-row .deps { color: var(--ink-soft); }
.plan-row.core { border-left: 3px solid var(--signal); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="masthead">
  <h1>Agentic LinkedIn Content Curator</h1>
  <p>topic → planner (DAG) → parallel search + image → generator → editor → post</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- API helpers
def _api_plan(topic: str) -> dict:
    resp = httpx.post(f"{API_BASE_URL}/plan", json={"topic": topic}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _parse_sse(raw_iter):
    """Minimal SSE line parser for sse_starlette's output -- no extra
    dependency for something this small. Yields {"event": <type>, **data}."""
    event_type, data_lines = None, []
    for line in raw_iter:
        if line == "":
            if event_type is not None:
                payload = json.loads("".join(data_lines)) if data_lines else {}
                yield {"event": event_type, **payload}
            event_type, data_lines = None, []
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())


def _api_revise(topic: str, post: dict) -> dict:
    resp = httpx.post(
        f"{API_BASE_URL}/revise", json={"topic": topic, "post": post}, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def _api_execute_stream(topic: str, plan: dict | None):
    payload = {"topic": topic}
    if plan is not None:
        payload["plan"] = plan
    with httpx.stream(
        "POST", f"{API_BASE_URL}/execute/stream", json=payload, timeout=httpx.Timeout(120.0)
    ) as resp:
        resp.raise_for_status()
        yield from _parse_sse(resp.iter_lines())


# ---------------------------------------------------------------- Gantt strip
def _client_waves(steps: list[dict]) -> dict[int, int]:
    """Topological level per step, computed from the plan the server just sent.

    The executor reports the same number in its final trace, but only at the
    END of the run -- without computing it here too, every step would render
    under "wave 0" for the whole execution and only snap into the right
    grouping once the result arrived, which looks like a bug even though the
    scheduling underneath is correct.
    """
    by_id = {s["step"]: s for s in steps}
    memo: dict[int, int] = {}

    def level(step_id: int) -> int:
        if step_id in memo:
            return memo[step_id]
        deps = [d for d in by_id[step_id]["depends_on"] if d in by_id]
        memo[step_id] = 0 if not deps else 1 + max(level(d) for d in deps)
        return memo[step_id]

    return {s["step"]: level(s["step"]) for s in steps}


def _render_strip(placeholder, step_state: dict[int, dict]) -> None:
    by_wave: dict[int, list[int]] = {}
    for sid, s in step_state.items():
        by_wave.setdefault(s.get("wave", 0), []).append(sid)

    cells = []
    for wave in sorted(by_wave):
        cells.append(f'<div class="stage wave-label"><div class="name">wave {wave}</div></div>')
        for sid in sorted(by_wave[wave]):
            s = step_state[sid]
            status = s["status"]
            ms = f'{s["ms"]} ms' if s.get("ms") is not None else ("running…" if status == "running" else "—")
            cells.append(
                f'<div class="stage {status}"><div class="name">{escape(s["tool"])}</div>'
                f'<div class="val">{escape(status)}</div><div class="ms">{escape(ms)}</div></div>'
            )
    placeholder.markdown(f'<div class="strip">{"".join(cells)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- input
with st.form("topic_form", clear_on_submit=False, border=False):
    col_a, col_b = st.columns([3, 1])
    with col_a:
        topic = st.text_input(
            "Topic",
            placeholder="recent trends in GenAI agents for backend engineers",
            label_visibility="collapsed",
        )
    with col_b:
        plan_clicked = st.form_submit_button("Plan it", width="stretch", type="primary")

st.caption(
    "Try: *recent trends in GenAI agents for backend engineers*  ·  "
    "*why I left consulting for a startup*  ·  *fundamentals of event-driven architecture*"
)

if plan_clicked and topic.strip():
    with st.spinner("Planning…"):
        try:
            plan_resp = _api_plan(topic.strip())
            st.session_state["topic"] = topic.strip()
            st.session_state["plan_resp"] = plan_resp
            st.session_state.pop("result", None)
        except httpx.TransportError:
            # Connection-level failure: the API isn't running (or isn't
            # reachable at this URL). Actionable -- say how to start it.
            st.error(API_DOWN_FIX)
        except httpx.HTTPError as e:
            # The API answered, but with an error status. Not fixable by
            # starting anything, so show what it actually said.
            st.error(f"The API returned an error: {e}")

plan_resp = st.session_state.get("plan_resp")

# ---------------------------------------------------------------- editable plan
if plan_resp:
    st.markdown("#### Plan")
    if plan_resp["used_llm_fallback"]:
        st.warning("The planner's own output was unusable; running the default plan instead.")
    if plan_resp["repair_notes"]:
        with st.expander(f"{len(plan_resp['repair_notes'])} plan repair(s) applied", expanded=False):
            for note in plan_resp["repair_notes"]:
                st.markdown(f"- {note}")

    steps = plan_resp["plan"]["steps"]
    included: dict[int, bool] = {}
    time_ranges: dict[int, str] = {}

    for s in sorted(steps, key=lambda x: x["step"]):
        is_core = s["tool"] in CORE_TOOLS
        cols = st.columns([0.6, 2, 3, 2] if not is_core else [0.6, 2, 5])
        with cols[0]:
            if is_core:
                st.markdown("**✓**")
                included[s["step"]] = True
            else:
                included[s["step"]] = st.checkbox(
                    "include", value=True, key=f"inc_{s['step']}", label_visibility="collapsed"
                )
        with cols[1]:
            st.markdown(f"`{s['tool']}`" + (" **(required)**" if is_core else ""))
        with cols[2]:
            deps = ", ".join(str(d) for d in s["depends_on"]) or "—"
            st.caption(f"step {s['step']} · depends on: {deps}")
        if not is_core:
            with cols[3]:
                if s["tool"] == "web_search":
                    options = ["day", "week", "month", "year", "(none)"]
                    current = s["args"].get("time_range") or "(none)"
                    if current not in options:
                        options.insert(0, current)
                    time_ranges[s["step"]] = st.selectbox(
                        "time_range", options, index=options.index(current),
                        key=f"tr_{s['step']}", label_visibility="collapsed",
                    )
                elif s["args"]:
                    st.caption(f"args: {json.dumps(s['args'])}")

    run_clicked = st.button("Run it", type="primary")

    if run_clicked:
        edited_steps = []
        for s in steps:
            if not included.get(s["step"], True):
                continue
            args = dict(s["args"])
            if s["step"] in time_ranges:
                tr = time_ranges[s["step"]]
                args["time_range"] = None if tr == "(none)" else tr
            edited_steps.append({**s, "args": args})
        edited_plan = {"steps": edited_steps}

        strip_placeholder = st.empty()
        step_state: dict[int, dict] = {}
        final_result = None

        try:
            for evt in _api_execute_stream(st.session_state["topic"], edited_plan):
                etype = evt.get("event")
                if etype == "planning":
                    strip_placeholder.info("Executing…")
                elif etype == "planned":
                    waves = _client_waves(evt["plan"]["steps"])
                    step_state = {
                        s["step"]: {
                            "tool": s["tool"],
                            "status": "pending",
                            "ms": None,
                            "wave": waves[s["step"]],
                        }
                        for s in evt["plan"]["steps"]
                    }
                    _render_strip(strip_placeholder, step_state)
                elif etype == "start":
                    if evt["step"] in step_state:
                        step_state[evt["step"]]["status"] = "running"
                    _render_strip(strip_placeholder, step_state)
                elif etype == "done":
                    if evt["step"] in step_state:
                        step_state[evt["step"]]["status"] = "ok" if evt["ok"] else "failed"
                        step_state[evt["step"]]["ms"] = evt["ms"]
                    _render_strip(strip_placeholder, step_state)
                elif etype == "result":
                    final_result = {k: v for k, v in evt.items() if k != "event"}
                    for st_trace in final_result["trace"]["steps"]:
                        if st_trace["step"] in step_state:
                            step_state[st_trace["step"]]["wave"] = st_trace["wave"]
                    _render_strip(strip_placeholder, step_state)
        except httpx.TransportError:
            # e.g. the API container was restarted mid-run.
            st.error(API_DOWN_FIX)
        except httpx.HTTPError as e:
            st.error(f"Execution failed: {e}")

        if final_result:
            st.session_state["result"] = final_result

# ---------------------------------------------------------------- result
result = st.session_state.get("result")
if result:
    post = result.get("post")
    image = result.get("image")
    trace = result["trace"]

    if trace["steps"]:
        degraded = set()
        for s in trace["steps"]:
            if not s["ok"]:
                degraded.add(s["tool"])
        if degraded and post and not post.get("refused"):
            st.warning(f"Degraded sources (proceeded anyway): {', '.join(sorted(degraded))}")

    col_post, col_img = st.columns([3, 2])

    with col_post:
        if post is None:
            st.error("No post was produced -- see the debug panel for the failure.")
        elif post.get("refused"):
            st.warning(
                "**The system refused to generate a post.**\n\n"
                f"Reason: {post.get('reason', 'insufficient source material')}"
            )
        else:
            tags = " ".join(f"#{escape(h)}" for h in post.get("hashtags", []))
            st.markdown(
                f'<div class="answer"><h3>{escape(post.get("headline", ""))}</h3>'
                f'<div class="body">{escape(post.get("body", ""))}</div>'
                f'<div class="tags">{tags}</div></div>',
                unsafe_allow_html=True,
            )
            if post.get("issues"):
                st.markdown("**Editor's critique**")
                for issue in post["issues"]:
                    st.markdown(f'<div class="issue-card">{escape(issue)}</div>', unsafe_allow_html=True)

            # The editor returns the pre-edit draft as `original`. Showing it is
            # the only way the UI can actually demonstrate what the critique
            # changed -- which is the entire reason the editor is a
            # critique-then-revise pass rather than a single silent rewrite.
            original = post.get("original")
            if original and original.get("headline"):
                with st.expander("Before / after — what the editor changed", expanded=False):
                    before, after = st.columns(2)
                    with before:
                        # After a "Revise again" press, `original` is the
                        # PREVIOUS revision, not the generator's first draft --
                        # each editor pass records the text it was handed. Label
                        # it for what it actually is.
                        prior = (
                            "previous version"
                            if st.session_state.get("revise_rounds")
                            else "generator's draft"
                        )
                        st.caption(f"BEFORE ({prior})")
                        st.markdown(
                            f'<div class="draft-card"><strong>{escape(original.get("headline", ""))}</strong>'
                            f'<div class="body">{escape(original.get("body", ""))}</div>'
                            f'<div class="tags">{" ".join("#" + escape(h) for h in original.get("hashtags", []))}</div>'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    with after:
                        st.caption("AFTER (editor's revision)")
                        st.markdown(
                            f'<div class="draft-card after"><strong>{escape(post.get("headline", ""))}</strong>'
                            f'<div class="body">{escape(post.get("body", ""))}</div>'
                            f'<div class="tags">{" ".join("#" + escape(h) for h in post.get("hashtags", []))}</div>'
                            "</div>",
                            unsafe_allow_html=True,
                        )

            # Re-runs ONLY the editor (POST /revise), so pressing it costs two
            # LLM calls and ~5s instead of re-running searches and regenerating
            # the image. Each press critiques the CURRENT text, so repeated
            # presses keep refining rather than re-deriving the same first edit.
            if st.button("Revise again", help="Re-run the Editor Agent over this post (no new search or image)"):
                with st.spinner("Editor reviewing again…"):
                    try:
                        revised = _api_revise(st.session_state["topic"], post)
                        if revised.get("post"):
                            st.session_state["result"]["post"] = revised["post"]
                            st.session_state["revise_rounds"] = st.session_state.get("revise_rounds", 0) + 1
                            st.rerun()
                    except httpx.TransportError:
                        st.error(API_DOWN_FIX)
                    except httpx.HTTPError as e:
                        st.error(f"Revision failed: {e}")

            if st.session_state.get("revise_rounds"):
                st.caption(f"editor passes: {1 + st.session_state['revise_rounds']}")

    with col_img:
        if image and image.get("image_b64"):
            st.image(base64.b64decode(image["image_b64"]), width="stretch")
            st.caption(f"tier: `{image['tier']}` · prompt: {image['prompt'][:120]}")
        else:
            st.info("No image produced.")

    # ------------------------------------------------------------ debug panel
    st.divider()
    with st.expander("Debug panel", expanded=True):
        tab_plan, tab_steps, tab_raw = st.tabs(["Plan", "Steps", "Raw trace"])

        with tab_plan:
            st.json(trace["plan"])
            if trace.get("planner_notes"):
                st.markdown("**Planner repair notes**")
                for note in trace["planner_notes"]:
                    st.markdown(f"- {note}")

        with tab_steps:
            for s in sorted(trace["steps"], key=lambda x: x["start_ms"]):
                icon = "✅" if s["ok"] else "❌"
                st.markdown(
                    f"{icon} **{s['tool']}** (step {s['step']}, wave {s['wave']}) — "
                    f"{s['start_ms']}–{s['end_ms']} ms"
                    + (f" — _{s['error']}_" if s.get("error") else "")
                )

        with tab_raw:
            st.json(trace)
