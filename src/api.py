"""
api.py — FastAPI REST server for AA Meetings RAG.

Run from the project root:
    uvicorn src.api:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure project root is on sys.path and .env is loaded ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.rag import async_run_rag
from src.vectorstore import (
    async_list_meetings,
    async_get_meeting_by_id,
    async_vector_search,
)
from src.embeddings import embed_query

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AA Meetings RAG API",
    description=(
        "Natural language search over AA meetings powered by "
        "Google Gemini + MongoDB Atlas Vector Search.\n\n"
        "**Quick start:** POST `/query` with a plain-English question."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_check():
    """Fail fast on missing config rather than getting cryptic errors at query time."""
    problems = []
    if not settings.gemini_api_key or "your_" in settings.gemini_api_key:
        problems.append("GEMINI_API_KEY not set in .env")
    if not settings.mongodb_uri or "your" in settings.mongodb_uri:
        problems.append("MONGODB_URI not set in .env")
    if settings.mongodb_uri.startswith("mongodb+srv://"):
        problems.append(
            "MONGODB_URI uses mongodb+srv:// — SRV DNS is blocked on this network. "
            "Run get_direct_uri.py and update .env with the direct mongodb:// URI."
        )
    if problems:
        for p in problems:
            print(f"[CONFIG ERROR] {p}")
        # Don't exit — let the server start so /health can report the issue


# ── Models ────────────────────────────────────────────────────────────────────

class QueryFilters(BaseModel):
    day: Optional[str] = Field(
        None, description="Day name ('Monday') or number (0=Sun … 6=Sat)"
    )
    region: Optional[str] = Field(
        None, description="Region name — must match exactly as stored e.g. 'LONG BEACH'"
    )
    types: Optional[List[str]] = Field(
        None, description="Meeting type codes e.g. ['BB', 'W', 'ONL']"
    )
    attendance_option: Optional[str] = Field(
        None, description="'in_person' | 'online' | 'hybrid'"
    )


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural language question",
        examples=["Are there Big Book meetings on Tuesday evenings in Long Beach?"],
    )
    filters: Optional[QueryFilters] = Field(
        None, description="Optional pre-filters applied before vector search"
    )
    top_k: Optional[int] = Field(
        None, ge=1, le=20, description="Meetings to retrieve (default: RAG_TOP_K in .env)"
    )


class QueryResponse(BaseModel):
    question: str                       # echo of the original question
    answer: str                         # Gemini answer — references meetings by [#N]
    meetings_cited: List[Dict[str, Any]] # only the meetings referenced in the answer
    total_cited: int                    # count of cited meetings


class VectorSearchRequest(BaseModel):
    query: str = Field(..., description="Text to embed and search")
    top_k: int = Field(8, ge=1, le=20)
    filters: Optional[QueryFilters] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"], summary="Health check")
async def health():
    return {
        "status": "ok",
        "gemini_model":     settings.gemini_model,
        "embedding_model":  settings.gemini_embedding_model,
        "mongodb_db":       settings.mongodb_db,
        "collection":       settings.mongodb_collection,
        "vector_index":     settings.vector_index_name,
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["rag"],
    summary="Natural language RAG query",
)
async def query_meetings(body: QueryRequest):
    """
    Embed the question → vector search Atlas → Gemini synthesises an answer.

    Optionally pass `filters` to narrow the vector search candidates before
    semantic ranking (e.g. restrict to a specific day or region).
    """
    filters_dict = _clean_filters(body.filters)
    try:
        answer, meetings = await async_run_rag(
            question=body.question,
            filters=filters_dict,
            top_k=body.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    serialised = [_serialise(m) for m in meetings]
    return QueryResponse(
        question=body.question,
        answer=answer,
        meetings_cited=serialised,
        total_cited=len(serialised),
    )


@app.post(
    "/search/vector",
    tags=["search"],
    summary="Raw vector search (no LLM)",
)
async def vector_search_endpoint(body: VectorSearchRequest):
    """
    Embed the query and return the closest meetings from Atlas Vector Search
    without passing them through Gemini. Useful for testing and custom UIs.
    """
    loop = asyncio.get_event_loop()
    try:
        query_vec = await loop.run_in_executor(None, embed_query, body.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    filters_dict = _clean_filters(body.filters)
    try:
        meetings = await async_vector_search(
            query_embedding=query_vec,
            top_k=body.top_k,
            filters=filters_dict,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "query":   body.query,
        "count":   len(meetings),
        "results": [_serialise(m) for m in meetings],
    }


@app.get(
    "/meetings",
    tags=["meetings"],
    summary="List / filter meetings (no vector search)",
)
async def list_meetings(
    day:         Optional[str] = Query(None, description="Day name or 0-6"),
    region:      Optional[str] = Query(None, description="Region e.g. LONG BEACH"),
    type:        Optional[str] = Query(None, description="Type code: BB, W, ONL, D, SP …"),
    time_of_day: Optional[str] = Query(None, description="morning | afternoon | evening | night"),
    attendance:  Optional[str] = Query(None, description="in_person | online | hybrid"),
    limit:       int           = Query(20, ge=1, le=100),
    skip:        int           = Query(0,  ge=0),
):
    """
    Simple filtered listing — no embeddings, no LLM.
    All parameters are optional and combinable.
    """
    try:
        meetings = await async_list_meetings(
            day=day,
            region=region,
            meeting_type=type,
            time_of_day=time_of_day,
            limit=limit,
            skip=skip,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"count": len(meetings), "meetings": [_serialise(m) for m in meetings]}


@app.get(
    "/meetings/{meeting_id}",
    tags=["meetings"],
    summary="Get meeting by ID",
)
async def get_meeting(meeting_id: str):
    """Fetch a single meeting by its `meeting_id` field."""
    try:
        doc = await async_get_meeting_by_id(meeting_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not doc:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found")

    return _serialise(doc)


@app.post(
    "/ingest/trigger",
    tags=["admin"],
    summary="Trigger re-ingestion from meetings.json",
)
async def trigger_ingest(
    background_tasks: BackgroundTasks,
    skip_inactive: bool = Query(True,  description="Skip inactive meetings"),
    limit:         int  = Query(0,     description="Max records (0 = all)"),
):
    """
    Re-run ingestion from the local `meetings.json` in the background.
    Returns immediately — check server logs for progress.
    """
    background_tasks.add_task(_bg_ingest, skip_inactive=skip_inactive, limit=limit)
    return {"message": "Ingestion started in background", "skip_inactive": skip_inactive, "limit": limit}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_filters(filters: Optional[QueryFilters]) -> Optional[Dict]:
    """Return a non-empty dict of filters, or None."""
    if not filters:
        return None
    d = {k: v for k, v in filters.model_dump().items() if v is not None}
    return d or None


def _serialise(doc: dict) -> dict:
    """Make a MongoDB document JSON-safe."""
    return {
        k: (str(v) if k == "_id" else v)
        for k, v in doc.items()
        if k not in ("embedding", "embedding_text", "_raw")
    }


async def _bg_ingest(skip_inactive: bool = True, limit: int = 0):
    """Background task: run ingest_local.py logic."""
    loop = asyncio.get_event_loop()

    def _run():
        # Import ingest_local as a module from the project root
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ingest_local", PROJECT_ROOT / "ingest_local.py"
        )
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run(
            file_path=str(PROJECT_ROOT / "meetings.json"),
            skip_inactive=skip_inactive,
            limit=limit,
        )

    await loop.run_in_executor(None, _run)
