"""
vectorstore.py — MongoDB Atlas Vector Search operations.

Provides both sync (ingest) and async (API) interfaces.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

from src.config import settings


# Add this to the top of ingest_local.py, before MongoClient is called
import pymongo
import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']

# ── SRV → standard URI resolver ───────────────────────────────────────────────

def _get_uri() -> str:
    """
    Return the MongoDB URI, raising a clear error if it's still an SRV URI
    (which requires DNS port 53 — often blocked on corporate/VPN/WSL networks).
    """
    uri = settings.mongodb_uri
    if uri.startswith("mongodb+srv://"):
        raise RuntimeError(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  mongodb+srv:// requires SRV DNS — blocked on your network  ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Run:  python get_direct_uri.py                              ║\n"
            "║  Or get the direct URI from Atlas:                           ║\n"
            "║    Cluster → Connect → Drivers → (non-SRV string)           ║\n"
            "║                                                              ║\n"
            "║  Then update .env:                                           ║\n"
            "║    MONGODB_URI=mongodb://user:pass@host1:27017,host2:27017/  ║\n"
            "║               ?ssl=true&replicaSet=...&authSource=admin      ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )
    return uri


# ── Sync client (reused across calls) ────────────────────────────────────────

_sync_client: Optional[MongoClient] = None

def get_sync_collection() -> Collection:
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(
            _get_uri(),
            serverSelectionTimeoutMS=10_000,
        )
    return _sync_client[settings.mongodb_db][settings.mongodb_collection]


# ── Async client (used by FastAPI) ────────────────────────────────────────────

_async_client = None

def get_async_collection():
    global _async_client
    if _async_client is None:
        from motor.motor_asyncio import AsyncIOMotorClient
        _async_client = AsyncIOMotorClient(
            _get_uri(),
            serverSelectionTimeoutMS=10_000,
        )
    return _async_client[settings.mongodb_db][settings.mongodb_collection]


# ── Vector search ─────────────────────────────────────────────────────────────

def vector_search(
    query_embedding: List[float],
    top_k: int = 8,
    filters: Optional[Dict[str, Any]] = None,
    collection: Optional[Collection] = None,
) -> List[Dict]:
    """
    Run Atlas $vectorSearch.
    filters: simple key→value dict, e.g. {"day": 1, "region": "LONG BEACH"}
    """
    col = collection or get_sync_collection()

    vector_stage: Dict[str, Any] = {
        "index":       settings.vector_index_name,
        "path":        "embedding",
        "queryVector": query_embedding,
        # numCandidates must be >= limit and <= 10000
        "numCandidates": max(top_k * 20, 150),
        "limit":       top_k,
    }

    # Only attach filter if non-empty — an empty filter object causes 0 results
    if filters:
        built = _build_filter(filters)
        if built:
            vector_stage["filter"] = built

    pipeline = [
        {"$vectorSearch": vector_stage},
        # Must use $addFields for $meta before an exclusion-only $project
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0, "embedding_text": 0, "_raw": 0}},
    ]

    try:
        return list(col.aggregate(pipeline))
    except Exception as exc:
        raise RuntimeError(
            f"$vectorSearch failed — check index name ({settings.vector_index_name!r}), "
            f"embedding dims, and that the index status is READY in Atlas.\n→ {exc}"
        ) from exc


async def async_vector_search(
    query_embedding: List[float],
    top_k: int = 8,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    col = get_async_collection()

    vector_stage: Dict[str, Any] = {
        "index":       settings.vector_index_name,
        "path":        "embedding",
        "queryVector": query_embedding,
        "numCandidates": max(top_k * 20, 150),
        "limit":       top_k,
    }

    if filters:
        built = _build_filter(filters)
        if built:
            vector_stage["filter"] = built

    pipeline = [
        {"$vectorSearch": vector_stage},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0, "embedding_text": 0, "_raw": 0}},
    ]

    try:
        cursor = col.aggregate(pipeline)
        return await cursor.to_list(length=top_k)
    except Exception as exc:
        raise RuntimeError(
            f"$vectorSearch failed — check index name ({settings.vector_index_name!r}), "
            f"embedding dims, and that the index status is READY in Atlas.\n→ {exc}"
        ) from exc


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_meetings(meetings: List[Dict], collection: Optional[Collection] = None):
    col = collection or get_sync_collection()
    ops = [
        UpdateOne({"meeting_id": m["meeting_id"]}, {"$set": m}, upsert=True)
        for m in meetings
        if m.get("meeting_id")
    ]
    if ops:
        return col.bulk_write(ops, ordered=False)
    return None


# ── Simple filtered list (no vector search) ───────────────────────────────────

async def async_list_meetings(
    day: Optional[str] = None,
    region: Optional[str] = None,
    meeting_type: Optional[str] = None,
    time_of_day: Optional[str] = None,
    limit: int = 20,
    skip: int = 0,
) -> List[Dict]:
    col = get_async_collection()
    query: Dict[str, Any] = {}

    if day:
        day_num = _day_name_to_num(day)
        query["day"] = day_num if day_num is not None else day
    if region:
        query["region"] = {"$regex": region, "$options": "i"}
    if meeting_type:
        query["types"] = {"$in": [meeting_type.upper()]}
    if time_of_day:
        tr = _time_of_day_range(time_of_day)
        if tr:
            query["time"] = {"$gte": tr[0], "$lt": tr[1]}

    cursor = col.find(query, {"embedding": 0, "embedding_text": 0}).skip(skip).limit(limit)
    return await cursor.to_list(length=limit)


async def async_get_meeting_by_id(meeting_id: str) -> Optional[Dict]:
    col = get_async_collection()
    return await col.find_one({"meeting_id": meeting_id}, {"embedding": 0})


# ── Filter builder ────────────────────────────────────────────────────────────

def _build_filter(filters: Dict[str, Any]) -> Dict:
    """
    Build a MongoDB $vectorSearch pre-filter from a simple key→value dict.

    IMPORTANT rules for Atlas vector pre-filters:
      - Only fields declared as 'filter' type in the index definition are allowed
      - $regex is NOT supported — use exact match or $in
      - Values must exactly match what's stored (case-sensitive for strings)
    """
    clauses = []
    for k, v in filters.items():
        if v is None:
            continue
        if k == "day":
            day_num = _day_name_to_num(v)
            val = day_num if day_num is not None else v
            clauses.append({"day": {"$eq": val}})
        elif k == "region":
            # Stored as uppercase e.g. "LONG BEACH" — normalise here
            clauses.append({"region": {"$eq": str(v).upper()}})
        elif k == "types":
            vals = [v] if isinstance(v, str) else list(v)
            clauses.append({"types": {"$in": vals}})
        elif k == "attendance_option":
            clauses.append({"attendance_option": {"$eq": v}})
        else:
            clauses.append({k: {"$eq": v}})

    if not clauses:
        return {}
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


# ── Day / time helpers ────────────────────────────────────────────────────────

_DAY_NAME_MAP = {
    "sunday": 0,    "sun": 0,
    "monday": 1,    "mon": 1,
    "tuesday": 2,   "tue": 2,  "tues": 2,
    "wednesday": 3, "wed": 3,
    "thursday": 4,  "thu": 4,  "thur": 4, "thurs": 4,
    "friday": 5,    "fri": 5,
    "saturday": 6,  "sat": 6,
}


def _day_name_to_num(day: Any) -> Optional[int]:
    if isinstance(day, int):
        return day
    if isinstance(day, str):
        return _DAY_NAME_MAP.get(day.lower().strip())
    return None


def _time_of_day_range(label: str):
    return {
        "morning":   ("06:00", "12:00"),
        "afternoon": ("12:00", "17:00"),
        "evening":   ("17:00", "21:00"),
        "night":     ("21:00", "24:00"),
    }.get(label.lower().strip())
