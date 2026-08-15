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
| **Intelligence** | LangGraph + Groq (Llama 3.3 70B) | 6-node graph; every node's output validated by Pydantic `with_structured_output`. |
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
| `GROQ_API_KEY` | ✅ | LLM inference (Llama 3.3 70B on Groq). |
| `TAVILY_API_KEY` | ✅ | Live web search. |
| `INTERNAL_API_KEY` | ✅ | Shared secret between the UI and API. Generate one with:<br>`python -c "import secrets; print(secrets.token_hex(32))"` |
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

# Full live end-to-end test (requires GROQ_API_KEY + TAVILY_API_KEY; spends quota)
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
rewrite. Model throughput (Groq free-tier daily token ceiling) is handled today via per-node
`max_tokens` budgets and is env-configurable for switching model tiers.

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
