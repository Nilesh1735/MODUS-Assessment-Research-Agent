"""
schemas.py
----------
Pydantic request/response models that form the FastAPI layer's public contract.

These are deliberately separate from the graph's internal ``ResearchResult`` dataclass: the
transport contract can evolve (add fields, versioning) without touching pipeline internals, and
the graph stays free of any web-framework types. All LLM/graph output is already Pydantic-
validated inside the pipeline; these models validate the HTTP boundary.
"""

from pydantic import BaseModel, Field, field_validator


class ResearchRequest(BaseModel):
    """Body of ``POST /api/research``."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The enterprise research question to investigate.",
        examples=["How is AI transforming retail?"],
    )

    @field_validator("question", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        # Trim before the length constraints apply, so a whitespace-only question is rejected.
        return value.strip() if isinstance(value, str) else value


class ResearchResponse(BaseModel):
    """Response of ``POST /api/research``."""

    session_id: int = Field(
        ...,
        description="Persisted research session id; use it to fetch this run's sources/findings.",
    )
    report: str = Field(
        ...,
        description="Final Markdown research report with inline [n] citations and a References section.",
    )
