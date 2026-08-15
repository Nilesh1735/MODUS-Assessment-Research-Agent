"""
graph.py
--------
Assemble the LangGraph research pipeline and expose a decoupled entry point.

Flow:

    START -> query_gen -> web_search -> grader --(conditional)--> extractor -> contradiction ->
             synthesizer -> END
                                       \\--(no relevant docs)--> synthesizer

The conditional edge after ``grader`` skips extraction/contradiction when nothing survived
relevance grading, routing straight to the synthesizer (which emits an "insufficient evidence"
report). This module imports nothing from the API/UI layers — it is intentionally standalone so
FastAPI, Streamlit, tests, or a CLI can all drive it the same way.

Observability: with the ``LANGCHAIN_*`` variables set in ``.env``, LangChain/LangGraph emit
LangSmith traces automatically — no tracing code is required here.
"""

import logging
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from backend.database import create_session, update_session_status
from backend.nodes.contradiction import contradiction
from backend.nodes.extractor import extractor
from backend.nodes.grader import grader
from backend.nodes.query_gen import query_gen
from backend.nodes.synthesizer import synthesizer
from backend.nodes.web_search import web_search
from backend.state import ResearchState

logger = logging.getLogger(__name__)

__all__ = ["ResearchState", "ResearchResult", "build_graph", "get_graph", "run_research"]


@dataclass(frozen=True)
class ResearchResult:
    """
    The return value of :func:`run_research`.

    ``session_id`` is the handle the API/UI layer uses to fetch this run's sources, findings,
    and contradictions (via the ``backend.database`` read helpers) to render alongside the
    report. Kept as a plain dataclass (not a Pydantic/API model) so the graph layer stays
    decoupled from any web framework; callers may serialise it however they like.
    """

    session_id: int
    report: str


def decide_after_grading(state: ResearchState) -> str:
    """Conditional edge: proceed to extraction only if any document passed grading."""
    if state.get("graded_documents"):
        return "extractor"
    logger.info("[graph] no relevant documents survived grading; routing grader -> synthesizer")
    return "synthesizer"


def build_graph():
    """Construct and compile the research ``StateGraph``."""
    builder = StateGraph(ResearchState)

    builder.add_node("query_gen", query_gen)
    builder.add_node("web_search", web_search)
    builder.add_node("grader", grader)
    builder.add_node("extractor", extractor)
    builder.add_node("contradiction", contradiction)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "query_gen")
    builder.add_edge("query_gen", "web_search")
    builder.add_edge("web_search", "grader")
    builder.add_conditional_edges(
        "grader",
        decide_after_grading,
        {"extractor": "extractor", "synthesizer": "synthesizer"},
    )
    builder.add_edge("extractor", "contradiction")
    builder.add_edge("contradiction", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()


_compiled_graph = None


def get_graph():
    """Return the process-wide compiled graph, building it on first use."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_research(question: str) -> ResearchResult:
    """
    Run the full pipeline for ``question`` and return the session id plus final Markdown report.

    Creates the session, invokes the graph, and marks the session 'failed' if the run raises.
    Intended as the single call the API/UI/tests use to trigger research. The returned
    ``session_id`` lets callers load the run's sources/findings/contradictions from the database.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    session_id = create_session(question)
    logger.info("[graph] starting research session=%s question=%r", session_id, question)
    try:
        final_state = get_graph().invoke({"question": question, "session_id": session_id})
    except Exception:
        update_session_status(session_id, "failed")
        logger.exception("[graph] research session=%s failed", session_id)
        raise

    return ResearchResult(session_id=session_id, report=final_state.get("report", ""))
