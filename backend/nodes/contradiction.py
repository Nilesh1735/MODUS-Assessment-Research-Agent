"""
nodes/contradiction.py
----------------------
Detect conflicting evidence among findings.

For each finding produced in this run, we query FAISS for its nearest neighbours (semantic
similarity), then ask the LLM whether any sufficiently-similar pair actually contradicts. Because
FAISS is a persistent, global index, neighbours may come from earlier sessions too — so the agent
detects contradictions across the whole accumulated corpus, not just within one run.

Similarity maths: embeddings are L2-normalised, and ``IndexFlatL2`` returns *squared* L2 distance
``D``. For unit vectors, cosine = 1 - D/2. We only LLM-check pairs whose cosine similarity clears
``CONTRADICTION_SIM_THRESHOLD``, and we never check the same pair twice.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend import config
from backend.database import faiss_store, get_finding_by_faiss_id, insert_contradiction
from backend.embeddings import embed
from backend.llm import ContradictionVerdict, structured_llm
from backend.state import ContradictionRecord, ResearchState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You compare two research findings and decide whether they genuinely contradict each other. "
    "A contradiction means the two statements cannot both be true (e.g. opposing facts, "
    "incompatible figures, or conflicting conclusions). Differences in scope, wording, level of "
    "detail, or complementary information are NOT contradictions. Be precise and conservative."
)


def _check_conflict(fact_a: str, fact_b: str) -> ContradictionVerdict:
    return structured_llm(ContradictionVerdict).invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Finding A:\n{fact_a}\n\n"
                    f"Finding B:\n{fact_b}\n\n"
                    "Do these two findings contradict each other?"
                )
            ),
        ]
    )


def contradiction(state: ResearchState) -> dict:
    """Contradiction node: findings -> logged conflicts (FAISS candidates + LLM verdict)."""
    session_id = state["session_id"]
    findings = state.get("findings", [])

    contradictions: list[ContradictionRecord] = []
    checked_pairs: set[tuple[int, int]] = set()

    for finding in findings:
        self_finding_id = finding["finding_id"]
        self_faiss_id = finding["faiss_id"]

        # +1 because the finding itself is in the index and will be its own nearest neighbour.
        distances, indices = faiss_store.search(
            embed(finding["fact"]), k=config.CONTRADICTION_TOP_K + 1
        )

        for distance, neighbour_faiss_id in zip(distances, indices):
            neighbour_faiss_id = int(neighbour_faiss_id)
            if neighbour_faiss_id < 0 or neighbour_faiss_id == self_faiss_id:
                continue

            cosine = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
            if cosine < config.CONTRADICTION_SIM_THRESHOLD:
                continue

            neighbour = get_finding_by_faiss_id(neighbour_faiss_id)
            if neighbour is None:
                continue
            other_finding_id = int(neighbour["id"])
            if other_finding_id == self_finding_id:
                continue

            pair = (min(self_finding_id, other_finding_id), max(self_finding_id, other_finding_id))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            try:
                verdict = _check_conflict(finding["fact"], neighbour["fact"])
            except Exception as exc:  # noqa: BLE001 - a failed check must not abort the node
                logger.warning("[contradiction] conflict check failed for pair %s: %s", pair, exc)
                continue

            if verdict.is_contradiction:
                insert_contradiction(
                    session_id=session_id,
                    finding_a_id=self_finding_id,
                    finding_b_id=other_finding_id,
                    explanation=verdict.explanation,
                    similarity=cosine,
                )
                contradictions.append(
                    ContradictionRecord(
                        finding_a_id=self_finding_id,
                        finding_b_id=other_finding_id,
                        explanation=verdict.explanation,
                        similarity=cosine,
                    )
                )

    logger.info(
        "[contradiction] session=%s: %d contradiction(s) across %d candidate pair(s)",
        session_id,
        len(contradictions),
        len(checked_pairs),
    )
    return {"contradictions": contradictions}
