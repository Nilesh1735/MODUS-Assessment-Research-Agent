"""
state.py
--------
The typed state that flows through the LangGraph pipeline.

It lives in its own module (rather than in ``graph.py``) so that individual node modules can
import the state type for annotations without creating a circular import back to ``graph.py``
(which itself imports the nodes). ``graph.py`` re-exports ``ResearchState`` for convenience.

Because the graph is a linear pipeline, each node returns a partial dict that LangGraph merges
into the running state; ``total=False`` reflects that keys are populated progressively.
"""

from typing import TypedDict


class Document(TypedDict):
    """A raw document retrieved from web search (before relevance grading)."""

    url: str
    title: str
    content: str
    query: str  # the sub-query that surfaced this document


class GradedDoc(TypedDict):
    """A document that passed relevance grading, with its persisted source id attached."""

    url: str
    title: str
    content: str
    query: str
    source_id: int


class FindingRecord(TypedDict):
    """An extracted finding after persistence to SQLite + FAISS."""

    finding_id: int
    source_id: int
    faiss_id: int
    fact: str
    classification: str
    confidence: float
    url: str


class ContradictionRecord(TypedDict):
    """A detected conflict between two findings."""

    finding_a_id: int
    finding_b_id: int
    explanation: str
    similarity: float


class ResearchState(TypedDict, total=False):
    """End-to-end state for one research run."""

    question: str                              # input: the research question
    session_id: int                            # input: the persisted session id
    sub_queries: list[str]                     # query_gen  -> web_search
    documents: list[Document]                  # web_search -> grader
    graded_documents: list[GradedDoc]          # grader     -> extractor
    findings: list[FindingRecord]              # extractor  -> contradiction / synthesizer
    contradictions: list[ContradictionRecord]  # contradiction -> synthesizer
    report: str                                # synthesizer (final output)
