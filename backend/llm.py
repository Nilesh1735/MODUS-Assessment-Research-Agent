"""
llm.py
------
Groq LLM factory and the Pydantic schemas that constrain every LLM output in the pipeline.

Why this shape:
  * The LLM is **lazily** constructed (``get_llm``) and cached, so importing the graph never
    requires an API key and never opens a network client until a node actually runs.
  * Every node that calls the model does so through ``with_structured_output(Schema)``. The
    provider is forced to return a tool call that validates against the schema, so nodes receive
    typed objects — never free-form text to regex/parse (a hard requirement of this project).
"""

import logging
import os
from functools import lru_cache
from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

from backend import config

logger = logging.getLogger(__name__)

# Classification vocabulary for extracted findings. A closed ``Literal`` set (rather than a free
# string) keeps downstream filtering/aggregation reliable and prevents label drift.
Classification = Literal[
    "fact",        # a stated, verifiable claim
    "statistic",   # a quantitative measure / figure
    "prediction",  # a forward-looking projection or forecast
    "opinion",     # a subjective assessment or expert view
    "definition",  # a conceptual explanation of a term
    "other",
]


# ─────────────────────────────────────────────────────────────────────────────
# LLM output schemas
# ─────────────────────────────────────────────────────────────────────────────

class SubQueries(BaseModel):
    """Decomposition of the research question into focused web-search sub-queries."""

    queries: list[str] = Field(
        ...,
        description=(
            f"Between {config.MIN_SUB_QUERIES} and {config.MAX_SUB_QUERIES} concise, diverse "
            "web-search queries that together cover the research question."
        ),
    )

    @field_validator("queries")
    @classmethod
    def _clean_and_bound(cls, value: list[str]) -> list[str]:
        # De-duplicate (case-insensitively), drop blanks, and cap at MAX_SUB_QUERIES.
        seen: set[str] = set()
        cleaned: list[str] = []
        for q in value:
            q = (q or "").strip()
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                cleaned.append(q)
        if not cleaned:
            # Raising triggers a structured-output retry; the node also has a fallback.
            raise ValueError("no non-empty sub-queries were produced")
        return cleaned[: config.MAX_SUB_QUERIES]


class GradedDocument(BaseModel):
    """Binary relevance judgement for a single retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="'yes' if the document is relevant to the research question, else 'no'.",
    )
    reason: str = Field(
        default="",
        description="One short sentence justifying the relevance decision.",
    )


class Finding(BaseModel):
    """A single atomic, self-contained finding extracted from a source document."""

    fact: str = Field(
        ...,
        description="A specific, standalone claim stated in full (no pronouns/references).",
    )
    classification: Classification = Field(
        default="fact",
        description="The type of claim this finding represents.",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="How strongly the source supports this finding, from 0.0 to 1.0.",
    )


class ExtractedFindings(BaseModel):
    """All findings extracted from one document."""

    findings: list[Finding] = Field(
        default_factory=list,
        description="Zero or more findings. Return an empty list if the document has no usable facts.",
    )


class ContradictionVerdict(BaseModel):
    """LLM verdict on whether two semantically-similar findings actually conflict."""

    is_contradiction: bool = Field(
        ...,
        description="True only if the two findings make claims that cannot both be true.",
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of the conflict, or why there is none.",
    )


class SynthesizedReport(BaseModel):
    """The final research report."""

    report_markdown: str = Field(
        ...,
        description=(
            "A well-structured Markdown report that answers the research question using ONLY "
            "the supplied findings, with inline bracketed citations like [1], [2] referencing "
            "the numbered sources, and a '## References' section listing those sources."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Build (once) and return the shared ChatGroq client.

    Raises a clear error if ``GROQ_API_KEY`` is missing, rather than failing deep inside a node.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env before running the research pipeline."
        )
    logger.info("Initialising ChatGroq (model=%s, temperature=%s)", config.GROQ_MODEL, config.LLM_TEMPERATURE)
    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=2,
    )


def structured_llm(schema: type[BaseModel]):
    """
    Return a runnable that invokes the LLM and yields a validated instance of ``schema``.

    Usage:  ``result: SubQueries = structured_llm(SubQueries).invoke(messages)``
    """
    return get_llm().with_structured_output(schema)
