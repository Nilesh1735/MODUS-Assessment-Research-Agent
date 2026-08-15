"""
nodes/query_gen.py
------------------
Entry node: decompose the user's research question into 3-4 focused web-search sub-queries.

The sub-queries are validated by the ``SubQueries`` schema and persisted to SQLite for
traceability. If structured generation fails for any reason, we fall back to searching the
raw question so the pipeline degrades gracefully instead of dead-ending.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend import config
from backend.database import insert_sub_queries, update_session_status
from backend.llm import SubQueries, structured_llm
from backend.state import ResearchState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert research strategist. Given a research question, decompose it into "
    f"{config.MIN_SUB_QUERIES}-{config.MAX_SUB_QUERIES} focused, self-contained web-search "
    "queries that together cover the distinct facets of the question (definitions, current "
    "state, evidence/statistics, challenges, and outlook where applicable). "
    "Each query must be concise and independently searchable. Do not number them."
)


def query_gen(state: ResearchState) -> dict:
    """LLM node: question -> validated list of sub-queries (persisted)."""
    question = state["question"]
    session_id = state["session_id"]
    logger.info("[query_gen] session=%s decomposing question", session_id)

    try:
        result = structured_llm(SubQueries, max_tokens=config.QUERY_GEN_MAX_TOKENS).invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Research question:\n{question}\n\n"
                        f"Return {config.MIN_SUB_QUERIES}-{config.MAX_SUB_QUERIES} search queries."
                    )
                ),
            ]
        )
        queries = result.queries
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any provider/validation error
        logger.warning(
            "[query_gen] structured generation failed (%s); falling back to raw question", exc
        )
        queries = [question]

    insert_sub_queries(session_id, queries)
    update_session_status(session_id, "searching")
    logger.info("[query_gen] session=%s produced %d sub-queries", session_id, len(queries))
    return {"sub_queries": queries}
