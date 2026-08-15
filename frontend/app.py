"""
frontend/app.py
---------------
Streamlit UI for the Modus Enterprise Research Agent.

This is a *thin client*: it never imports the pipeline or the database. It sends the research
question to the FastAPI backend over HTTP (``POST /api/research`` with an ``X-API-Key`` header)
and renders the returned Markdown report. Keeping the UI decoupled from the graph means the two
can be deployed, scaled, and secured independently.

Run with:  ``streamlit run frontend/app.py``  (with the API running via ``uvicorn backend.main:app``)
"""

import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_API_KEY = os.getenv("INTERNAL_API_KEY", "")
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "300"))

EXAMPLE_QUESTIONS = [
    "How is AI transforming retail?",
    "What are the enterprise risks of large language models?",
    "How are companies adopting renewable energy in supply chains?",
]

st.set_page_config(page_title="Modus Research Agent", page_icon="🔎", layout="wide")

# ── Light enterprise styling ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 1100px; }
      .app-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0; }
      .app-subtitle { color: #6b7280; font-size: 1.02rem; margin-top: .2rem; }
      div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Persist the last result so it survives Streamlit reruns (e.g. clicking Download).
st.session_state.setdefault("result", None)
st.session_state.setdefault("elapsed", None)
st.session_state.setdefault("question", "")


# ── Sidebar: connection + diagnostics ────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Connection")

    # Both the API base URL and the API key are read silently from the environment (.env) and are
    # never rendered as inputs — nothing environment-specific or sensitive appears on screen.
    api_url = DEFAULT_API_URL.rstrip("/")
    api_key = DEFAULT_API_KEY
    if not api_key:
        st.error("INTERNAL_API_KEY is not set in `.env`. Add it and restart the app to run research.")

    if st.button("Check backend health", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/api/health", timeout=10)
            if r.ok:
                h = r.json()
                st.success("Backend reachable.")
                st.json(h)
            else:
                st.error(f"Health check failed: HTTP {r.status_code}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Cannot reach backend at {api_url}\n\n{exc}")

    st.divider()
    st.caption(
        "Pipeline: **query decomposition → web search → relevance grading → finding extraction "
        "→ contradiction detection → synthesis**. Every finding is stored with its source, and "
        "the report cites sources inline as `[n]`."
    )


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🔎 Enterprise Research Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask a research question and receive a structured, '
    "citation-traceable report synthesised from live web sources.</div>",
    unsafe_allow_html=True,
)
st.write("")

# Example prompts (set the input before the text area is instantiated this run).
st.write("**Try an example:**")
ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
for i, example in enumerate(EXAMPLE_QUESTIONS):
    if ex_cols[i].button(example, key=f"ex_{i}", use_container_width=True):
        st.session_state.question = example

question = st.text_area(
    "Research question",
    key="question",
    height=110,
    placeholder="e.g. How is AI transforming retail?",
)

run_clicked = st.button("🚀 Run Research", type="primary", use_container_width=True)


# ── Run ──────────────────────────────────────────────────────────────────────
if run_clicked:
    q = (question or "").strip()
    if len(q) < 3:
        st.warning("Please enter a research question (at least 3 characters).")
    elif not api_key:
        st.error("INTERNAL_API_KEY is not set in `.env`. Add it and restart the app.")
    else:
        with st.spinner(
            "Running multi-step research (search → grade → extract → detect contradictions → "
            "synthesise). This can take 1–3 minutes…"
        ):
            try:
                t0 = time.time()
                resp = requests.post(
                    f"{api_url}/api/research",
                    json={"question": q},
                    headers={"X-API-Key": api_key},
                    timeout=REQUEST_TIMEOUT,
                )
                elapsed = time.time() - t0
            except requests.exceptions.Timeout:
                st.error(f"The request timed out after {REQUEST_TIMEOUT:.0f}s. Try again or raise API_TIMEOUT_SECONDS.")
                st.stop()
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the backend at {api_url}. Is `uvicorn backend.main:app` running?\n\n{exc}")
                st.stop()

        if resp.status_code == 200:
            st.session_state.result = resp.json()
            st.session_state.elapsed = elapsed
        elif resp.status_code == 401:
            st.session_state.result = None
            st.error("Authentication failed (401). The INTERNAL_API_KEY in this app's `.env` does not match the server's key.")
        elif resp.status_code == 429:
            st.warning("Rate limit exceeded (429). Please wait a moment and try again.")
        elif resp.status_code == 422:
            detail = resp.json().get("detail", "Invalid request.")
            st.error(f"Invalid request (422): {detail}")
        else:
            st.error(f"Backend error (HTTP {resp.status_code}): {resp.text[:400]}")


# ── Result (rendered from session state so it persists across reruns) ─────────
result = st.session_state.result
if result:
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Session ID", f"#{result['session_id']}")
    if st.session_state.elapsed is not None:
        c2.metric("Generated in", f"{st.session_state.elapsed:.0f}s")

    with st.container(border=True):
        st.markdown(result["report"])

    st.download_button(
        "⬇️ Download report (.md)",
        data=result["report"],
        file_name=f"research_session_{result['session_id']}.md",
        mime="text/markdown",
        use_container_width=True,
    )
