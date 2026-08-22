"""
tests/test_pipeline_smoke.py
----------------------------
Smoke tests for the research pipeline.

Two tiers:
  * Structural tests run everywhere with no API keys — they prove every module imports, the
    graph compiles, and the Pydantic schemas enforce their contracts.
  * The live end-to-end test runs the real pipeline against Groq + Tavily and is skipped unless
    both API keys are present. It also serves as the persistence check: run it twice and the
    FAISS vector count and prior sessions should carry over (data persists across restarts).
"""

import os

import pytest

from backend.database import (
    faiss_store,
    get_connection,
    get_findings,
    init_db,
)

LIVE_KEYS_PRESENT = bool(os.getenv("OPENROUTER_API_KEY")) and bool(os.getenv("TAVILY_API_KEY"))
LIVE_QUESTION = "How is AI transforming retail?"


# ── Structural tests (no API keys required) ──────────────────────────────────

def test_graph_compiles():
    """The full StateGraph must import and compile without any API keys or network access."""
    from backend.graph import build_graph

    compiled = build_graph()
    assert hasattr(compiled, "invoke"), "compiled graph should expose an invoke() method"


def test_subqueries_schema_bounds():
    """SubQueries de-duplicates, drops blanks, and caps the number of queries."""
    from backend import config
    from backend.llm import SubQueries

    result = SubQueries(queries=["a", "a", " ", "b", "c", "d", "e", "f"])
    assert len(result.queries) <= config.MAX_SUB_QUERIES
    assert "a" in result.queries and result.queries.count("a") == 1

    with pytest.raises(ValueError):
        SubQueries(queries=["   ", ""])


def test_finding_confidence_is_bounded():
    """Finding.confidence must stay within [0, 1]."""
    from backend.llm import Finding

    assert Finding(fact="x", classification="fact", confidence=0.5).confidence == 0.5
    with pytest.raises(ValueError):
        Finding(fact="x", classification="fact", confidence=1.5)


def test_contradiction_verdict_coercion():
    """ContradictionVerdict uses a yes/no enum (not a bool) and exposes is_contradiction.

    Regression guard: a raw ``bool`` field caused Groq to reject the tool call server-side
    (``expected boolean, but got string``). The enum removes that failure mode; the ``mode=before``
    normaliser is defense-in-depth for any non-tool-calling path.
    """
    from backend.llm import ContradictionVerdict

    assert ContradictionVerdict(verdict="yes").is_contradiction is True
    assert ContradictionVerdict(verdict="no").is_contradiction is False

    # Normaliser maps booleans and common variants onto the enum.
    assert ContradictionVerdict(verdict=True).verdict == "yes"
    assert ContradictionVerdict(verdict=False).verdict == "no"
    assert ContradictionVerdict(verdict="FALSE").is_contradiction is False
    assert ContradictionVerdict(verdict="No").is_contradiction is False

    # The generated schema exposes a string enum, not a boolean; is_contradiction is a property.
    schema = ContradictionVerdict.model_json_schema()
    assert schema["properties"]["verdict"]["enum"] == ["yes", "no"]
    assert "is_contradiction" not in schema["properties"]

    with pytest.raises(ValueError):
        ContradictionVerdict(verdict="maybe")


def test_db_helpers_roundtrip():
    """The additive DB helpers persist and read back correctly (no LLM involved)."""
    from backend.database import (
        create_session,
        insert_finding,
        insert_source,
        insert_sub_queries,
        update_session_status,
    )

    init_db()
    session_id = create_session("unit-test question")
    assert isinstance(session_id, int)

    insert_sub_queries(session_id, ["q1", "q2"])
    source_id = insert_source(session_id, "https://example.com", "Example", "body", 1.0)
    finding_id = insert_finding(session_id, source_id, "a fact", "fact", 0.9, faiss_index_id=0)

    rows = get_findings(session_id)
    assert any(r["id"] == finding_id and r["url"] == "https://example.com" for r in rows)

    update_session_status(session_id, "completed")
    with get_connection() as conn:
        status = conn.execute(
            "SELECT status FROM research_sessions WHERE id = ?", (session_id,)
        ).fetchone()["status"]
    assert status == "completed"


def test_research_result_shape():
    """run_research's return type carries both the session id and the report."""
    from backend.graph import ResearchResult

    result = ResearchResult(session_id=7, report="# report")
    assert result.session_id == 7
    assert result.report == "# report"


# ── Live end-to-end test (requires OPENROUTER_API_KEY + TAVILY_API_KEY) ──────

@pytest.mark.skipif(not LIVE_KEYS_PRESENT, reason="OPENROUTER_API_KEY and TAVILY_API_KEY required")
def test_run_research_end_to_end():
    """Run the real pipeline via run_research and assert a traceable, persisted result."""
    from backend.graph import run_research

    init_db()
    vectors_before = faiss_store.total_vectors

    result = run_research(LIVE_QUESTION)
    assert result.session_id > 0
    assert result.report, "pipeline must return a non-empty report"
    assert "## References" in result.report, "report must contain a References section"

    # Findings were extracted and persisted, and the report row was written.
    assert len(get_findings(result.session_id)) > 0, "expected at least one persisted finding"
    with get_connection() as conn:
        report_row = conn.execute(
            "SELECT content FROM reports WHERE session_id = ?", (result.session_id,)
        ).fetchone()
    assert report_row is not None and report_row["content"] == result.report

    # FAISS grew — vectors persist to disk across restarts.
    assert faiss_store.total_vectors > vectors_before
