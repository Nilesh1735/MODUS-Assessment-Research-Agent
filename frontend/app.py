"""
frontend/app.py
---------------
Streamlit UI for the Enterprise AI Research Agent.

Thin HTTP client: it never imports the pipeline or the database. It sends the research question to
the FastAPI backend over HTTP (POST /api/research with an X-API-Key header) and renders the returned
Markdown report. The UI stays decoupled from the graph so the two can be deployed and scaled apart.

Design language mirrors the "Dark Emerald / Illuminated" aesthetic: Space Grotesk display type,
an emerald accent (#10b981 -> #059669), soft illumination, and a centred enterprise-SaaS layout.

Run with:  streamlit run frontend/app.py   (with the API up via  uvicorn backend.main:app)
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

st.set_page_config(page_title="Enterprise AI Research Agent", layout="wide")


# ── Theme: Dark Emerald / Illuminated ─────────────────────────────────────────
# Centralised CSS injection. Hides default Streamlit chrome and restyles the app into a
# premium, glowing enterprise surface. No f-string here, so literal CSS braces are safe.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

      :root {
        --bg-0: #06090f;
        --bg-1: #0a0e15;
        --surface: #0f1520;
        --surface-2: #131b28;
        --border: rgba(16, 185, 129, 0.16);
        --border-soft: rgba(255, 255, 255, 0.07);
        --text: #e9eff0;
        --text-muted: #8b97a5;
        --accent: #10b981;
        --accent-2: #059669;
        --accent-bright: #34d399;
        --glow: rgba(16, 185, 129, 0.35);
      }

      /* Hide default Streamlit chrome: header, footer, hamburger menu, toolbar, decoration bar. */
      header[data-testid="stHeader"] { display: none; }
      [data-testid="stToolbar"] { display: none; }
      [data-testid="stDecoration"] { display: none; }
      [data-testid="stStatusWidget"] { display: none; }
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; height: 0; }

      /* App background: deep near-black with an emerald illumination glow at the top. */
      .stApp {
        background:
          radial-gradient(1000px 520px at 50% -8%, rgba(16, 185, 129, 0.10), transparent 60%),
          linear-gradient(180deg, var(--bg-0), var(--bg-1) 60%);
        background-attachment: fixed;
        color: var(--text);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }

      /* Centred, max-width enterprise layout. */
      .block-container {
        max-width: 1000px;
        margin: 0 auto;
        padding-top: 3rem;
        padding-bottom: 4rem;
      }

      h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--text); }

      /* ── Hero ─────────────────────────────────────────────────────────── */
      .hero { text-align: center; margin-bottom: 2rem; }
      .hero-tag {
        display: inline-block;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent-bright);
        border: 1px solid var(--border);
        background: rgba(16, 185, 129, 0.06);
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        margin-bottom: 1.1rem;
      }
      .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.6rem;
        line-height: 1.1;
        margin: 0;
        color: #f4f8f7;
        text-shadow: 0 0 42px rgba(16, 185, 129, 0.28);
      }
      .hero-title .lit {
        background: linear-gradient(120deg, var(--accent-bright), var(--accent));
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }
      .hero-sub {
        color: var(--text-muted);
        font-size: 1.05rem;
        margin-top: 0.75rem;
        max-width: 640px;
        margin-left: auto; margin-right: auto;
      }
      .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 0.4rem 0 0.6rem 0.2rem;
      }

      /* ── Bordered containers (input + output) ─────────────────────────── */
      [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, var(--surface), var(--surface-2));
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        padding: 1.4rem 1.5rem !important;
        box-shadow: 0 24px 60px -32px rgba(0, 0, 0, 0.9), inset 0 1px 0 rgba(255, 255, 255, 0.03);
      }

      /* ── Text area ────────────────────────────────────────────────────── */
      .stTextArea textarea {
        background: #0b111a;
        color: var(--text);
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }
      .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.18);
      }
      .stTextArea textarea::placeholder { color: #5c6675; }
      .stTextArea label { color: var(--text-muted) !important; font-weight: 500; }

      /* ── Buttons: base ────────────────────────────────────────────────── */
      .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        transition: transform 0.18s ease, box-shadow 0.2s ease,
                    background 0.2s ease, border-color 0.2s ease;
      }

      /* Secondary buttons (examples, health check, download): subtle glass. */
      .stButton > button[kind="secondary"],
      button[data-testid="stBaseButton-secondary"],
      .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.03);
        color: var(--text-muted);
        border: 1px solid var(--border-soft);
      }
      .stButton > button[kind="secondary"]:hover,
      button[data-testid="stBaseButton-secondary"]:hover,
      .stDownloadButton > button:hover {
        color: var(--text);
        border-color: var(--accent);
        background: rgba(16, 185, 129, 0.08);
        transform: translateY(-1px);
        box-shadow: 0 10px 26px -14px var(--glow);
      }

      /* Primary button (Run Research): full-width illuminated emerald. */
      .stButton > button[kind="primary"],
      button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--accent-bright), var(--accent-2));
        color: #04130d;
        border: none;
        padding: 0.7rem 1rem;
        font-size: 1.02rem;
        box-shadow: 0 12px 34px -10px var(--glow);
      }
      .stButton > button[kind="primary"]:hover,
      button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 46px -10px var(--glow), 0 0 0 1px rgba(52, 211, 153, 0.5);
        filter: brightness(1.04);
      }
      .stButton > button[kind="primary"]:active,
      button[data-testid="stBaseButton-primary"]:active { transform: translateY(0); }

      /* ── Metrics as cards ─────────────────────────────────────────────── */
      [data-testid="stMetric"] {
        background: linear-gradient(180deg, var(--surface), var(--surface-2));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 16px 40px -28px rgba(0, 0, 0, 0.9);
      }
      [data-testid="stMetricLabel"] p {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem !important;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-muted) !important;
      }
      [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--accent-bright);
      }

      /* ── Report typography inside the output container ────────────────── */
      .stMarkdown a { color: var(--accent-bright); text-decoration: none; }
      .stMarkdown a:hover { text-decoration: underline; }
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #f2f7f5; }
      .stMarkdown h2 { border-bottom: 1px solid var(--border-soft); padding-bottom: 0.35rem; }
      .stMarkdown code { font-family: 'Space Mono', monospace; color: var(--accent-bright); }
      .stMarkdown blockquote {
        border-left: 3px solid var(--accent);
        background: rgba(16, 185, 129, 0.05);
        border-radius: 0 8px 8px 0;
      }

      /* ── Sidebar ──────────────────────────────────────────────────────── */
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080c12, #0a0f17);
        border-right: 1px solid var(--border-soft);
      }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { font-size: 1.1rem; }

      /* Emerald-tint the loading spinner. */
      .stSpinner > div > div { border-top-color: var(--accent) !important; }

      /* Divider + scrollbar polish. */
      hr { border-color: var(--border-soft); }
      ::-webkit-scrollbar { width: 10px; height: 10px; }
      ::-webkit-scrollbar-thumb { background: rgba(16, 185, 129, 0.25); border-radius: 8px; }
      ::-webkit-scrollbar-track { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Persist the last result so it survives Streamlit reruns (e.g. clicking Download).
st.session_state.setdefault("result", None)
st.session_state.setdefault("elapsed", None)
st.session_state.setdefault("question", "")

api_url = DEFAULT_API_URL.rstrip("/")
api_key = DEFAULT_API_KEY


# ── Sidebar: status + diagnostics only (no URL / key inputs) ──────────────────
with st.sidebar:
    st.header("System")
    st.caption(
        "The API base URL and key are read silently from the environment (.env) and never "
        "rendered as inputs, so nothing sensitive appears on screen."
    )

    if api_key:
        st.markdown(
            "<span style='color:#34d399;font-family:Space Mono,monospace;font-size:.8rem;'>"
            "&#9679; Credentials loaded</span>",
            unsafe_allow_html=True,
        )
    else:
        st.error("INTERNAL_API_KEY is not set in .env. Add it and restart the app to run research.")

    st.divider()

    if st.button("Check backend health", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/api/health", timeout=10)
            if r.ok:
                st.success("Backend reachable.")
                st.json(r.json())
            else:
                st.error(f"Health check failed: HTTP {r.status_code}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Cannot reach backend at {api_url}\n\n{exc}")

    st.divider()
    st.caption(
        "Pipeline: query decomposition, web search, relevance grading, finding extraction, "
        "contradiction detection, and synthesis. Every finding is stored with its source, and the "
        "report cites sources inline as [n]."
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <div class="hero-tag">Enterprise AI &middot; Secured API</div>
      <h1 class="hero-title">Enterprise <span class="lit">AI</span> Research Agent</h1>
      <p class="hero-sub">
        Ask a research question and receive a structured, citation-traceable report
        synthesised from live web sources by a multi-step reasoning pipeline.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Example prompts (prefill the input before the text area is instantiated) ──
st.markdown('<div class="section-label">Try one of these</div>', unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
for i, example in enumerate(EXAMPLE_QUESTIONS):
    if ex_cols[i].button(example, key=f"ex_{i}", use_container_width=True):
        st.session_state.question = example


# ── Input surface ─────────────────────────────────────────────────────────────
with st.container(border=True):
    question = st.text_area(
        "Research question",
        key="question",
        height=120,
        placeholder="e.g. How is AI transforming retail?",
    )
    run_clicked = st.button("Run Research", type="primary", use_container_width=True)


# ── Run ────────────────────────────────────────────────────────────────────────
if run_clicked:
    q = (question or "").strip()
    if len(q) < 3:
        st.warning("Please enter a research question (at least 3 characters).")
    elif not api_key:
        st.error("INTERNAL_API_KEY is not set in .env. Add it and restart the app.")
    else:
        with st.spinner("Running multi-step research pipeline (this takes 1-3 minutes)..."):
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
                st.error(f"Could not reach the backend at {api_url}. Is uvicorn backend.main:app running?\n\n{exc}")
                st.stop()

        if resp.status_code == 200:
            st.session_state.result = resp.json()
            st.session_state.elapsed = elapsed
        elif resp.status_code == 401:
            st.session_state.result = None
            st.error("Authentication failed (401). The INTERNAL_API_KEY in this app's .env does not match the server's key.")
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
        c2.metric("Generation Time", f"{st.session_state.elapsed:.1f}s")

    st.markdown('<div class="section-label">Research report</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(result["report"])

    st.download_button(
        "Download report (.md)",
        data=result["report"],
        file_name=f"research_session_{result['session_id']}.md",
        mime="text/markdown",
        use_container_width=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center;color:#5c6675;font-family:Space Mono,monospace;"
    "font-size:.74rem;letter-spacing:.06em;margin-top:2.5rem;'>"
    "Decoupled LangGraph pipeline &middot; Authenticated API &middot; Citations traceable to source"
    "</p>",
    unsafe_allow_html=True,
)
