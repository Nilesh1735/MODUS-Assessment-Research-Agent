"""
nodes/grader.py
---------------
Grade each retrieved document for relevance to the research question and filter out the junk.

Traceability choice: *every* collected document is persisted to ``sources`` with its relevance
score (1.0 for relevant, 0.0 for filtered), so the audit trail records exactly what the agent
saw and how it judged each item. Only relevant documents — carrying their new ``source_id`` —
are forwarded to the extractor.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend import config
from backend.database import insert_source
from backend.llm import GradedDocument, structured_llm
from backend.state import Document, GradedDoc, ResearchState

logger = logging.getLogger(__name__)

# A short excerpt is enough to judge relevance; keep the grading prompt cheap.
_GRADE_EXCERPT_CHARS = 1500

_SYSTEM_PROMPT = (
    "You are a strict relevance grader for an enterprise research agent. Decide whether the "
    "document contains information useful for answering the research question. Grade 'yes' only "
    "if it is genuinely on-topic and substantive; grade 'no' for tangential, promotional, or "
    "empty content. Be conservative."
)


def _grade_document(question: str, doc: Document) -> GradedDocument:
    excerpt = (doc["content"] or "")[:_GRADE_EXCERPT_CHARS]
    return structured_llm(GradedDocument, max_tokens=config.GRADER_MAX_TOKENS).invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Research question:\n{question}\n\n"
                    f"Document title: {doc['title']}\n"
                    f"Document excerpt:\n{excerpt}\n\n"
                    "Is this document relevant?"
                )
            ),
        ]
    )


def grader(state: ResearchState) -> dict:
    """Grading node: raw documents -> persisted sources + relevant documents only."""
    session_id = state["session_id"]
    question = state["question"]
    documents = state.get("documents", [])

    relevant: list[GradedDoc] = []
    for doc in documents:
        try:
            verdict = _grade_document(question, doc)
            is_relevant = verdict.binary_score == "yes"
        except Exception as exc:  # noqa: BLE001 - fail closed (exclude) but keep going
            logger.warning("[grader] grading failed for %s: %s", doc.get("url"), exc)
            is_relevant = False

        source_id = insert_source(
            session_id=session_id,
            url=doc["url"],
            title=doc["title"],
            raw_content=doc["content"],
            relevance_score=1.0 if is_relevant else 0.0,
        )

        if is_relevant:
            relevant.append(
                GradedDoc(
                    url=doc["url"],
                    title=doc["title"],
                    content=doc["content"],
                    query=doc["query"],
                    source_id=source_id,
                )
            )

    logger.info(
        "[grader] session=%s: %d/%d documents graded relevant",
        session_id,
        len(relevant),
        len(documents),
    )
    return {"graded_documents": relevant}
