"""
main.py
-------
FastAPI application that exposes the research pipeline over authenticated HTTP.

Design:
  * **Auth** — every ``/api/research`` call must present a valid ``X-API-Key`` header matching
    ``INTERNAL_API_KEY``. The comparison is constant-time (``secrets.compare_digest``), and the
    endpoint fails *closed* (503) if the server has no key configured — it never serves research
    unauthenticated because of a misconfiguration.
  * **Decoupling** — this module imports ``run_research`` from the graph and nothing else from the
    pipeline internals. The graph knows nothing about FastAPI; this layer knows nothing about
    LangGraph. Either can be replaced independently.
  * **Non-blocking** — ``run_research`` is synchronous and long-running (many LLM/network calls),
    so it is dispatched to a worker thread via ``run_in_threadpool`` to keep the event loop free.
  * **Rate limiting** — slowapi caps how often a client may trigger the expensive research run.

Run with:  ``uvicorn backend.main:app --reload``
"""

import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend import config  # noqa: F401 - imported for its load_dotenv() side effect
from backend.database import init_db
from backend.graph import run_research
from backend.schemas import ResearchRequest, ResearchResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How often a single client may hit the research endpoint. Overridable via env for deployment.
API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10/minute")
API_KEY_HEADER = "X-API-Key"

# auto_error=False lets us return our own 401 and distinguish "missing key" from "wrong key".
_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def require_api_key(provided_key: str | None = Depends(_api_key_header)) -> None:
    """FastAPI dependency: enforce a valid ``X-API-Key`` header (constant-time comparison)."""
    expected = os.getenv("INTERNAL_API_KEY")
    if not expected:
        # Fail closed: a server with no configured key must not answer authenticated routes.
        logger.error("INTERNAL_API_KEY is not configured; rejecting request.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server authentication is not configured.",
        )
    if not provided_key or not secrets.compare_digest(provided_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the SQLite schema exists before the API accepts traffic."""
    init_db()
    logger.info("Database initialised; research API ready.")
    yield


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Modus Enterprise AI Research Agent API",
    version="1.0.0",
    description="Authenticated API for running structured, citation-traceable enterprise research.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/api/health", tags=["ops"])
async def health() -> dict:
    """Unauthenticated liveness/readiness probe.

    Reports whether required secrets are present as booleans only — never the values themselves.
    """
    return {
        "status": "ok",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
        "auth_configured": bool(os.getenv("INTERNAL_API_KEY")),
    }


@app.post(
    "/api/research",
    response_model=ResearchResponse,
    dependencies=[Depends(require_api_key)],
    tags=["research"],
)
@limiter.limit(API_RATE_LIMIT)
async def research(request: Request, payload: ResearchRequest) -> ResearchResponse:
    """Run the full research pipeline for a question and return its session id + Markdown report.

    ``request`` is required by slowapi for per-client limiting; ``payload`` is the validated body.
    """
    logger.info("Research request accepted (question length=%d).", len(payload.question))
    try:
        result = await run_in_threadpool(run_research, payload.question)
    except ValueError as exc:  # empty/blank question that slipped past schema validation
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception:  # noqa: BLE001 - surface a clean error; details stay in server logs
        logger.exception("Research run failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research run failed. Check the server logs for details.",
        )
    return ResearchResponse(session_id=result.session_id, report=result.report)
