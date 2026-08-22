# 🔎 Enterprise AI Research Agent

An end-to-end, enterprise-grade research agent built for the **Modus Enterprise AI Build Challenge**.
Ask any research question and receive a structured, **citation-traceable** Markdown report synthesised
from live web sources — with every finding persisted to disk and every claim traceable back to a URL.

The system runs a **decoupled, multi-node LangGraph pipeline** behind an **authenticated FastAPI API**,
with a thin **Streamlit** front end. All LLM output is validated with Pydantic structured output — there
is no `exec`/`eval`, no regex/string-parsing of model output, and no single "giant prompt".

---

## Architecture

Five real layers, each independently deployable, scalable, and replaceable:

```
┌──────────────┐     HTTP  (X-API-Key header)      ┌───────────────────────────────┐
│   Streamlit  │ ────────────────────────────────▶ │      FastAPI  (backend)       │
│  (frontend)  │ ◀──────────────────────────────── │  auth · rate-limit · REST API │
└──────────────┘    JSON {session_id, report}      └───────────────┬───────────────┘
   thin client                                                      │ run_research(question)
   (no pipeline imports)                                            ▼
                                              ┌────────────────────────────────────────┐
                                              │           LangGraph pipeline            │
                                              │  query_gen → web_search → grader →      │
                                              │  extractor → contradiction → synthesizer│
                                              └───────┬──────────────────────┬──────────┘
                                                      │                      │
                                              ┌───────▼────────┐   ┌─────────▼───────────────┐
                                              │   Tavily API   │   │     SQLite + FAISS      │
                                              │ (live web      │   │  sessions · sub_queries │
                                              │  search)       │   │  sources · findings ·   │
                                              └────────────────┘   │  contradictions ·       │
                                                                    │  reports · vectors      │
                                                                    └─────────────────────────┘
```

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| **Frontend** | Streamlit | Thin HTTP client — sends the question, renders the report. No pipeline/DB imports. |
| **Backend** | FastAPI + slowapi | Authenticated (`X-API-Key`, constant-time), rate-limited REST API. |
| **Intelligence** | LangGraph + Google Gemini API (Gemini 3.5 Flash-Lite) | 6-node graph; every node's output validated by Pydantic `with_structured_output`. |
| **Data** | SQLite (WAL) + FAISS | Durable relational store + vector index for semantic contradiction detection. |
| **External** | Tavily | Live web search. |

### The pipeline (what each node does)

1. **query_gen** — decomposes the question into 3–4 focused sub-queries.
2. **web_search** — runs each sub-query through Tavily, de-duplicates by URL.
3. **grader** — scores each document's relevance (binary yes/no); persists every source.
4. **extractor** — extracts atomic findings, classifies each, embeds them into FAISS.
5. **contradiction** — searches FAISS for semantically-similar findings and asks the LLM whether they conflict.
6. **synthesizer** — writes the final Markdown report with inline `[n]` citations; the **References
   section is appended deterministically in code**, so citations always resolve to real URLs.

---

## Setup

Requires **Python 3.11+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Configure secrets (`.env`)

```bash
# Copy the template and fill in your keys
cp .env.template .env
```

Then edit `.env`:

| Variable | Required | Notes |
|----------|----------|-------|
| `GEMINI_API_KEY` | ✅ | LLM inference (Google Gemini API). |
| `TAVILY_API_KEY` | ✅ | Live web search. |
| `INTERNAL_API_KEY` | ✅ | Shared secret between the UI and API. Generate one with:<br>`python -c "import secrets; print(secrets.token_hex(32))"` |
| `GROQ_MODEL` | optional | Gemini model id to use. Default `gemini-3.5-flash-lite`; change to swap models. |
| `LANGCHAIN_API_KEY` / `LANGCHAIN_TRACING_V2` / `LANGCHAIN_PROJECT` | optional | LangSmith tracing. |
| `API_BASE_URL` | optional | Where the UI reaches the API. Default `http://127.0.0.1:8000`. |
| `API_TIMEOUT_SECONDS` | optional | UI request timeout. Default `300`. |
| `API_RATE_LIMIT` | optional | Per-client limit on `/api/research`. Default `10/minute`. |
| `DATABASE_PATH` / `FAISS_INDEX_PATH` | optional | Where state is persisted. |

> **Security:** `.env` is git-ignored and never committed. The API key is read silently from the
> environment — the UI has no input box for it, so it can never be exposed on screen during a demo.
> The API compares keys in constant time (`secrets.compare_digest`) and **fails closed** (503) if no
> key is configured.

---

## Running the app

Open **two terminals** (both with the virtual environment activated).

**Terminal 1 — API:**
```bash
uvicorn backend.main:app
```
> On Windows, omit `--reload` (its file-watcher can abort on start); restart manually after edits.

**Terminal 2 — UI:**
```bash
streamlit run frontend/app.py
```

Open the Streamlit URL (default http://localhost:8501), use **"Check backend health"** in the sidebar
to confirm connectivity, then ask a question. Reports can be downloaded as Markdown.

The interactive API docs are at http://127.0.0.1:8000/docs.

---

## Testing

```bash
# Fast, keyless structural tests (no API quota spent)
pytest tests/test_pipeline_smoke.py -k "not end_to_end"

# Full live end-to-end test (requires GEMINI_API_KEY + TAVILY_API_KEY; spends quota)
pytest tests/test_pipeline_smoke.py
```

The live test asserts that a real run produces a report **and that findings persist across a restart**
(new DB connection + reloaded FAISS index).

---

## Data persistence

State survives restarts by design:

- **SQLite** (`research.db`, WAL mode) stores sessions, sub-queries, sources, findings, contradictions,
  and reports. Reports are upserted (`ON CONFLICT`) so re-running a session is idempotent.
- **FAISS** (`faiss_index`) stores finding embeddings and is reloaded from disk on startup.

Both paths are git-ignored so local state never leaks into the repository.

---

## Data model

All research state lives in **SQLite (six tables)** and is mirrored into a **FAISS** vector index.
Every table links back to a `research_sessions` row, so one run — question, sub-queries, sources,
findings, contradictions, and report — is a single connected graph keyed on `session_id`.

| Table | Key columns | Links to | Purpose |
|-------|-------------|----------|---------|
| `research_sessions` | `id`, `question`, `status`, `created_at` | — | One row per research run. |
| `sub_queries` | `id`, `session_id`, `query` | → `research_sessions` | The decomposed search queries. |
| `sources` | `id`, `session_id`, `url`, `title`, `raw_content`, `relevance_score` | → `research_sessions` | Every web document collected (relevant or not). |
| `findings` | `id`, `session_id`, `source_id`, `fact`, `classification`, `confidence`, `faiss_index_id` | → `research_sessions`, `sources` | Atomic extracted claims; `faiss_index_id` maps to the vector store. |
| `contradictions` | `id`, `session_id`, `finding_a_id`, `finding_b_id`, `explanation`, `similarity` | → `research_sessions`, `findings` ×2 | Detected conflicts between two findings. |
| `reports` | `id`, `session_id` (UNIQUE), `content` | → `research_sessions` | Final Markdown report (upserted, so re-runs are idempotent). |

**Vector store.** FAISS (`IndexFlatL2`, 384-dim, cosine via L2-normalised vectors) holds one vector
per finding. The link is bidirectional: `findings.faiss_index_id` stores the vector's row id, and
`get_finding_by_faiss_id()` reverse-maps a FAISS hit back to its finding — this is how the
contradiction node turns a semantic-similarity neighbour into a concrete, cited claim.

**Traceability chain:** report `[n]` → `findings.source_id` → `sources.url`. Every sentence in a
report walks back to the exact URL it came from.

---

## The "1,000 records tomorrow" question

**Honest answer:** the current design is correct and durable for the assignment's scale (a single
analyst asking questions), and the *decoupling* is what makes scaling a config/infra swap rather than a
rewrite. But scaling to 1,000 concurrent research jobs would expose three real bottlenecks, and I'd
address them in this order:

1. **Request model → async job queue.** Today `/api/research` is a synchronous 1–3 minute call that
   holds a threadpool worker for the entire run (≈40 concurrent before requests serialize). At scale I'd
   return `202 Accepted` + a `job_id`, run the pipeline on a worker pool (arq / Celery / RQ), and add a
   `GET /api/research/{job_id}` status/poll endpoint.

2. **Vector store → a networked, concurrent index.** The FAISS layer is a single-process, in-memory
   `IndexFlatL2` that rewrites the whole file on every add (O(n²) I/O) and does brute-force O(n) search.
   A `threading.Lock` currently makes it *correct* for the single-process deployment, but it cannot scale
   horizontally. At scale I'd move vectors to **pgvector / Qdrant / Weaviate** (concurrent, networked,
   ANN indexes like HNSW/IVF), key vectors on the stable SQLite finding id, and scope contradiction
   search per session with a metadata filter instead of scanning the global corpus.

3. **Relational store → Postgres.** SQLite is single-writer; under concurrent load it serializes writes
   (mitigated here with `busy_timeout` + WAL). At scale I'd move to Postgres for true concurrent writers,
   keeping the same schema and helper functions.

Because the graph, API, and data layers are already fully decoupled (the graph imports nothing from
FastAPI or Streamlit), none of the above touches the node logic — it's an infrastructure swap, not a
rewrite. Model throughput (Gemini free-tier rate limits) is handled today via per-node
`max_tokens` budgets and is env-configurable for switching model tiers.

---

## External services & free-tier contingency

Only **two** operations leave the machine — LLM inference and web search. Everything else
(embeddings, vector index, relational store, API, UI) runs locally with no key and no external
dependency. Both external services are on free tiers today, and both are deliberately isolated so
they can be swapped — or replaced with a fully local alternative — without touching the graph.

**Google Gemini API (LLM inference) — free tier, generous per-minute limits.**
The model is reached only through `get_llm()` in [backend/llm.py](backend/llm.py); the six nodes are
model-agnostic (they only call `structured_llm(schema)`). The client is a stock
`ChatGoogleGenerativeAI` from the `langchain-google-genai` package. If it becomes paid or unavailable:
- switch models with the `GROQ_MODEL` env var, or switch providers by swapping the chat-model class
  in [backend/llm.py](backend/llm.py) for any LangChain chat integration (OpenAI, Anthropic,
  Together, Groq) — without editing a single node; **or**
- run **fully local — no key, no cost** — with Ollama or vLLM serving Llama 3.x / Qwen. The
  structured-output contract is identical.

**Tavily (web search) — free tier (~1,000 searches/month).**
Exactly one node, [backend/nodes/web_search.py](backend/nodes/web_search.py), talks to Tavily. If it
becomes paid or unavailable, replace that single client with the Brave Search API, SerpAPI, the
keyless `duckduckgo-search` package, or a self-hosted SearXNG. Nothing else in the graph, data
layer, or report logic changes.

**Bottom line:** there is no lock-in. The daily free quota is stretched by the per-node `max_tokens`
budgets in [backend/config.py](backend/config.py), and either external service can be moved to a
local, zero-cost implementation.

---

## Models & libraries and their licences

Everything used is free and open-source (permissive licences) or a free-tier API — no paid or
commercial licence is required to build or run this project.

**Models**

| Model | Role | How it runs | Licence |
|-------|------|-------------|---------|
| `gemini-3.5-flash-lite` | Reasoning, extraction, synthesis | Google Gemini API (free tier) | Proprietary (free tier) |
| `BAAI/bge-small-en-v1.5` | Sentence embeddings for similarity/contradiction | **Locally**, via `fastembed` (ONNX runtime, no API call) | MIT |

**Core libraries**

| Library | Role | Licence |
|---------|------|---------|
| fastapi | API framework | MIT |
| uvicorn | ASGI server | BSD-3-Clause |
| streamlit | Frontend | Apache-2.0 |
| langchain / langgraph | LLM orchestration + stateful graph | MIT |
| langchain-google-genai | Google Gemini chat-model binding | MIT |
| langchain-community | Community integrations | MIT |
| tavily-python | Web-search client | MIT |
| faiss-cpu | Vector index | MIT |
| fastembed | Local ONNX embedding runtime | Apache-2.0 |
| pydantic | Structured-output validation | MIT |
| slowapi | Rate limiting | MIT |
| python-dotenv | Env loading | BSD-3-Clause |
| numpy | Numerics | BSD-3-Clause |
| requests / httpx | HTTP clients | Apache-2.0 / BSD-3-Clause |
| langsmith | Optional tracing | MIT |
| pytest / pytest-asyncio | Tests | MIT / Apache-2.0 |

All bundled libraries and the local embedding model use permissive licences (MIT / BSD / Apache-2.0).
The LLM (Gemini 3.5 Flash-Lite) is a proprietary model consumed via the Google Gemini API's **free tier** — no
weights are bundled in this repo, and the factory can be repointed at an open model (Llama 3.x / Qwen via
Ollama) with no changes outside [backend/llm.py](backend/llm.py).

---

## AI assistance disclosure

As required by the challenge, this discloses where AI coding tools were used.

AI coding assistants (Anthropic Claude / Claude Code) were used during development to accelerate
boilerplate (FastAPI wiring, Streamlit layout, SQL helpers), to draft this documentation, and for
code review and refactoring suggestions. **All architectural decisions were made and reviewed by the
author** — the decoupled layer boundaries, the six-node LangGraph topology, the SQLite + FAISS data
model, the authentication / rate-limiting design, and the structured-output-validation strategy.
Every file in this repository is understood and explainable by the author; nothing is included that
cannot be walked through and justified.

---

## Design guarantees (why this isn't a "ChatGPT wrapper")

- **Multi-step reasoning graph**, not one giant prompt — 6 distinct, single-responsibility nodes.
- **All LLM output is Pydantic-validated** (`with_structured_output`) — no regex, no `json.loads`,
  no brittle string parsing.
- **No arbitrary code execution** — no `exec`, `eval`, `subprocess`, `os.system`, or `pickle` anywhere.
- **Authenticated API** — constant-time `X-API-Key`, fail-closed, rate-limited.
- **Full traceability** — every finding links to a source row; the report cites `[n]` inline and the
  References section is generated in code so citations can't drift.
- **Decoupled layers** — UI ↔ API ↔ graph ↔ data are independent and separately testable.

---

## Challenge compliance

**Mandatory five-layer architecture** — all present (see the diagram above): UI (Streamlit) ·
API (FastAPI) · AI intelligence (LangGraph + Google Gemini) · data & knowledge (SQLite + FAISS) ·
external research (Tavily).

| Requirement | Where it's met |
|-------------|----------------|
| Working app: real frontend, backend, data, AI integration | This repo — all four layers run. |
| Data persists across restarts | SQLite (WAL) + FAISS on disk; asserted by the live test. |
| Multiple records processed systematically, not hard-coded | Any question is a "record"; the same six nodes run on whatever is asked. |
| Outputs are traceable | Inline `[n]` citations → `sources.url`; deterministic References section. |
| Not a notebook / slideshow / no-code tool | Python packages: a FastAPI service + a Streamlit app. |
| Free / open-source / free-tier only | See licences above; free-tier contingency documented. |
| Architecture diagram | "Architecture" section. |
| Database / data model | "Data model" section. |
| Model & library inventory with licences | "Models & libraries and their licences" section. |
| Setup instructions | "Setup" + "Running the app". |
| AI-tool usage disclosed | "AI assistance disclosure" section. |
| Scale to 1,000 records/day | "The 1,000 records tomorrow" section. |

**Records & sample data.** This is a live-research agent, so it ships no static dataset — each
research *question* is a record, and running one populates SQLite + FAISS with real sources and
findings. Example questions are provided in the UI and this README, and any new question works out
of the box (the "surprise record" test): enter it, watch the six-node pipeline run, inspect the
cited report.

**Assignment 9 step mapping.** query_gen (define questions) → web_search (search sources) → grader
(collect / keep relevant) + `sources` table (store sources) → extractor (extract + classify
findings) → contradiction (compare evidence via FAISS, detect conflicts) → synthesizer (conclusions
with traceable citations). The persisted SQLite + FAISS store **is** the reusable knowledge base:
contradiction detection searches *all* prior findings, so knowledge accumulates across sessions.
