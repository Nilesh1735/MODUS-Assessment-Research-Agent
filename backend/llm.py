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
    """
    LLM verdict on whether two semantically-similar findings actually conflict.

    The verdict is a closed ``Literal["yes", "no"]`` string, not a raw ``bool``. Groq validates
    tool-call arguments against the generated JSON schema *server-side* and rejects mismatches with
    a 400 before the response reaches us — and tool-calling models frequently emit ``"false"`` (a
    string) for a boolean field, which triggers exactly that rejection. A closed enum sidesteps the
    problem (mirroring the proven :class:`GradedDocument` pattern); ``is_contradiction`` remains
    available as a computed property so call sites are unaffected.
    """

    verdict: Literal["yes", "no"] = Field(
        ...,
        description="'yes' if the two findings make claims that cannot both be true, else 'no'.",
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of the conflict, or why there is none.",
    )

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, value: object) -> object:
        # Defense-in-depth for any non-tool-calling path (e.g. JSON mode): coerce booleans and
        # common variants onto the enum. On the tool-calling path Groq validates the enum itself.
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"yes", "true", "y", "contradiction", "contradictory"}:
                return "yes"
            if v in {"no", "false", "n", "none", "consistent"}:
                return "no"
        return value

    @property
    def is_contradiction(self) -> bool:
        """True when the two findings genuinely conflict."""
        return self.verdict == "yes"


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

@lru_cache(maxsize=8)
def get_llm(max_tokens: int) -> ChatGroq:
    """
    Build (once per ``max_tokens`` budget) and return a shared ChatGroq client.

    ``max_tokens`` is part of the cache key because Groq counts reserved output tokens against the
    per-minute limit, so nodes request only the budget they need (see ``structured_llm``). A small
    set of budgets means only a handful of cached clients. Raises a clear error if ``GROQ_API_KEY``
    is missing, rather than failing deep inside a node.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env before running the research pipeline."
        )
    logger.info(
        "Initialising ChatGroq (model=%s, temperature=%s, max_tokens=%s)",
        config.GROQ_MODEL,
        config.LLM_TEMPERATURE,
        max_tokens,
    )
    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=max_tokens,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=config.LLM_MAX_RETRIES,
    )


def structured_llm(schema: type[BaseModel], max_tokens: int | None = None):
    """
    Return a runnable that invokes the LLM and yields a validated instance of ``schema``.

    ``max_tokens`` bounds this call's reserved output (defaults to ``config.LLM_MAX_TOKENS``); pass
    a per-node budget to keep small calls from tripping Groq's per-minute token limit.

    Usage:  ``result: SubQueries = structured_llm(SubQueries, max_tokens=512).invoke(messages)``
    """
    resolved = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
    return get_llm(resolved).with_structured_output(schema)
