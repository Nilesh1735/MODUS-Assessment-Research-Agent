"""
nodes/extractor.py
------------------
Extract structured findings from each relevant document, then persist them to SQLite and index
their embeddings in FAISS.

Ordering matters: this node fully populates both ``findings`` and the FAISS index before the
contradiction node runs, so intra-run (and cross-run) similarity search has every new finding
available. Each finding's fact is embedded (normalised) and added to FAISS; the returned vector
id is stored on the finding row as ``faiss_index_id`` to keep SQLite <-> FAISS linked.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend import config
from backend.database import faiss_store, insert_finding
from backend.embeddings import embed
from backend.llm import ExtractedFindings, structured_llm
from backend.state import FindingRecord, GradedDoc, ResearchState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise information-extraction engine for an enterprise research agent. "
    "From the document, extract atomic, standalone findings that are relevant to the research "
    "question. Each finding must be a single, self-contained claim written in full (resolve "
    "pronouns and references so it stands alone). Classify each finding and assign a confidence "
    "in [0,1] reflecting how strongly the document supports it. Extract only what the document "
    "actually states — never infer or invent. If there is nothing useful, return an empty list."
)


def _extract_from_document(question: str, doc: GradedDoc) -> ExtractedFindings:
    content = (doc["content"] or "")[: config.MAX_CONTENT_CHARS]
    return structured_llm(ExtractedFindings, max_tokens=config.EXTRACTOR_MAX_TOKENS).invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Research question:\n{question}\n\n"
                    f"Source title: {doc['title']}\n"
                    f"Source URL: {doc['url']}\n"
                    f"Document content:\n{content}"
                )
            ),
        ]
    )


def extractor(state: ResearchState) -> dict:
    """Extraction node: relevant documents -> persisted findings (SQLite + FAISS)."""
    session_id = state["session_id"]
    question = state["question"]
    graded_documents = state.get("graded_documents", [])

    findings: list[FindingRecord] = []
    for doc in graded_documents:
        try:
            extracted = _extract_from_document(question, doc)
        except Exception as exc:  # noqa: BLE001 - skip a bad document, keep the run alive
            logger.warning("[extractor] extraction failed for %s: %s", doc["url"], exc)
            continue

        for item in extracted.findings:
            fact = item.fact.strip()
            if not fact:
                continue
            try:
                faiss_id = faiss_store.add(embed(fact))
                finding_id = insert_finding(
                    session_id=session_id,
                    source_id=doc["source_id"],
                    fact=fact,
                    classification=item.classification,
                    confidence=item.confidence,
                    faiss_index_id=faiss_id,
                )
            except Exception as exc:  # noqa: BLE001 - skip a single finding on persistence error
                logger.warning("[extractor] failed to persist finding %r: %s", fact[:80], exc)
                continue

            findings.append(
                FindingRecord(
                    finding_id=finding_id,
                    source_id=doc["source_id"],
                    faiss_id=faiss_id,
                    fact=fact,
                    classification=item.classification,
                    confidence=item.confidence,
                    url=doc["url"],
                )
            )

    logger.info(
        "[extractor] session=%s extracted %d findings from %d documents",
        session_id,
        len(findings),
        len(graded_documents),
    )
    return {"findings": findings}
