"""
embeddings.py — Gemini embedding wrapper using the google-genai SDK.

Model: gemini-embedding-001  (3072 dims, supports task_type hints)
"""
import time
from typing import List

from google import genai
from google.genai import types as genai_types
from google.api_core import exceptions as google_exceptions
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception, before_sleep_log,
)
import logging

from src.config import settings

log = logging.getLogger(__name__)

# ── Single shared client ───────────────────────────────────────────────────────
_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


# ── Task type constants ────────────────────────────────────────────────────────
TASK_RETRIEVAL_DOCUMENT = "retrieval_document"
TASK_RETRIEVAL_QUERY    = "retrieval_query"


def _is_retryable(exc: BaseException) -> bool:
    """
    Only retry on transient server/network errors — NOT on 4xx client errors.
    A RetryError wrapping a ClientError (400/403/404) means bad key, quota, or
    wrong model name; retrying those just wastes time and hides the real message.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, google_exceptions.GoogleAPICallError):
        # 5xx = transient server error → retry
        # 4xx = our fault (bad key, quota, wrong model) → raise immediately
        return exc.grpc_status_code is None or getattr(exc, "code", 500) >= 500
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def embed_text(text: str, task_type: str = TASK_RETRIEVAL_QUERY) -> List[float]:
    """Embed a single string. Returns a list of floats (3072 dims)."""
    try:
        result = _get_client().models.embed_content(
            model=settings.gemini_embedding_model,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type=task_type),
        )
        return result.embeddings[0].values
    except Exception as exc:
        # Unwrap and re-raise with a clear message so the CLI shows the real cause
        msg = str(exc)
        if "API_KEY" in msg.upper() or "api key" in msg.lower():
            raise RuntimeError(f"Gemini API key invalid or missing. Check GEMINI_API_KEY in .env\n→ {msg}") from exc
        if "not found" in msg.lower() or "not supported" in msg.lower():
            raise RuntimeError(
                f"Embedding model '{settings.gemini_embedding_model}' not found. "
                f"Try 'gemini-embedding-001' in .env GEMINI_EMBEDDING_MODEL\n→ {msg}"
            ) from exc
        if "quota" in msg.lower() or "429" in msg:
            raise RuntimeError(f"Gemini quota exceeded. Wait and retry.\n→ {msg}") from exc
        raise


def embed_document(text: str) -> List[float]:
    return embed_text(text, task_type=TASK_RETRIEVAL_DOCUMENT)


def embed_query(text: str) -> List[float]:
    return embed_text(text, task_type=TASK_RETRIEVAL_QUERY)


def embed_batch(texts: List[str], task_type: str = TASK_RETRIEVAL_DOCUMENT) -> List[List[float]]:
    embeddings = []
    for i, text in enumerate(texts):
        embeddings.append(embed_text(text, task_type=task_type))
        if (i + 1) % 100 == 0:
            time.sleep(1)
    return embeddings


def build_document_text(meeting: dict) -> str:
    parts = []

    name = meeting.get("name", "")
    if name:
        parts.append(f"Meeting: {name}")

    location = meeting.get("location", "") or meeting.get("location_text", "")
    if location:
        parts.append(f"Location: {location}")

    address = meeting.get("formatted_address", "") or " ".join(
        filter(None, [
            meeting.get("address", ""),
            meeting.get("city", ""),
            meeting.get("state", ""),
            meeting.get("postal_code", ""),
        ])
    )
    if address.strip():
        parts.append(f"Address: {address}")

    day = meeting.get("day_text", "") or _day_num_to_text(meeting.get("day"))
    time_str = meeting.get("time_formatted", "") or meeting.get("time", "")
    if day:
        parts.append(f"Day: {day}")
    if time_str:
        parts.append(f"Time: {time_str}")

    region = meeting.get("region", "")
    if region:
        parts.append(f"Region: {region}")

    types = meeting.get("types", [])
    if types:
        type_labels = _expand_types(types)
        parts.append(f"Meeting types: {', '.join(type_labels)}")

    attendance = meeting.get("attendance_option", "")
    if attendance:
        parts.append(f"Attendance: {attendance.replace('_', ' ')}")

    notes = (meeting.get("notes", "") or "").strip()
    if notes:
        parts.append(f"Notes: {notes[:300]}")

    return ". ".join(parts)


_DAY_MAP = {
    0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
    4: "Thursday", 5: "Friday", 6: "Saturday",
}

_TYPE_MAP = {
    "12x12": "12 Steps & 12 Traditions", "1HR": "One Hour Meeting",
    "A": "Atheist/Agnostic", "ABSI": "As Bill Sees It",
    "ASL": "American Sign Language", "B": "Beginners",
    "BE": "Beginner/Newcomer", "BS": "Book Study", "C": "Closed",
    "CAN": "Candlelight", "CF": "Child Friendly", "D": "Discussion",
    "DR": "Daily Reflections", "EN": "En Español", "InP": "In Person",
    "LGBTQ": "LGBTQ+", "LIT": "Literature", "LS": "Living Sober",
    "M": "Men", "MED": "Meditation", "NB": "Non-Binary", "O": "Open",
    "ONL": "Online/Virtual", "Part": "Participation", "QA": "Question & Answer",
    "S": "Spanish", "SEN": "Seniors", "SP": "Speaker", "SS": "Step Study",
    "ST": "Step", "T": "Traditions", "TC": "Temporary Closed Contact",
    "VM": "Virtual Meeting", "W": "Women", "X": "Open to All",
    "XT": "Open to All (extended)", "Y": "Young People",
}


def _day_num_to_text(day) -> str:
    if day is None:
        return ""
    try:
        return _DAY_MAP.get(int(day), "")
    except (TypeError, ValueError):
        return str(day)


def _expand_types(types) -> List[str]:
    if isinstance(types, str):
        types = [t.strip() for t in types.split(",")]
    return [_TYPE_MAP.get(t, t) for t in types if t]
