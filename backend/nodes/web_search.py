"""
nodes/web_search.py
-------------------
Search the web (via Tavily) for each sub-query and collect the raw documents into state.

No database writes happen here: relevance is unknown until the grader runs, so persistence of
``sources`` is deferred to that node. Results are de-duplicated by URL, and long page bodies are
truncated to keep downstream token usage bounded.
"""

import logging
import os
from functools import lru_cache

from backend import config
from backend.state import Document, ResearchState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_tavily_client():
    """Build (once) and return the Tavily client, importing the SDK lazily."""
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env before running the research pipeline."
        )
    return TavilyClient(api_key=api_key)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit]


def web_search(state: ResearchState) -> dict:
    """Search node: sub-queries -> de-duplicated raw documents."""
    session_id = state["session_id"]
    sub_queries = state.get("sub_queries") or [state["question"]]
    client = get_tavily_client()

    documents: list[Document] = []
    seen_urls: set[str] = set()

    for query in sub_queries:
        try:
            response = client.search(
                query=query,
                max_results=config.TAVILY_MAX_RESULTS,
                search_depth=config.TAVILY_SEARCH_DEPTH,  # type: ignore[arg-type]
                include_raw_content=True,
            )
        except Exception as exc:  # noqa: BLE001 - one failed query must not abort the run
            logger.warning("[web_search] query %r failed: %s", query, exc)
            continue

        for result in response.get("results", []):
            url = result.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            raw = result.get("raw_content") or result.get("content") or ""
            documents.append(
                Document(
                    url=url,
                    title=result.get("title") or url,
                    content=_truncate(raw, config.MAX_CONTENT_CHARS),
                    query=query,
                )
            )

    logger.info(
        "[web_search] session=%s collected %d unique documents across %d sub-queries",
        session_id,
        len(documents),
        len(sub_queries),
    )
    return {"documents": documents}
