"""
rag.py — Core RAG chain.

The LLM is instructed to reference meetings by [#N] markers.
run_rag() parses those markers and returns only the cited meetings,
in citation order — so the CLI table and API JSON match exactly
what the answer text references.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _clean_answer(text: str) -> str:
    """Remove [#N] citation markers from the answer before returning to the user."""
    return re.sub(r'\[#\d+\]\s*', '', text).strip()

from google import genai as google_genai
from google.genai import types as genai_types

from config import settings
from embeddings import embed_query
from vectorstore import vector_search, async_vector_search

# ── Gemini client ─────────────────────────────────────────────────────────────

_llm_client = google_genai.Client(api_key=settings.gemini_api_key)

_SYSTEM_INSTRUCTION = """\
You are a helpful AA meetings assistant. You will be given a numbered list of
meetings retrieved from a database. Use ONLY those meetings to answer.

Rules:
- Reference every meeting you mention with its number in brackets e.g. [#1] [#3].
- List each relevant meeting: name, day, time, location, and types.
- If none match the question, say so and suggest broadening the search.
- Never invent details not present in the context.
- Stay on topic — AA meetings only.
"""


# ── Public API ────────────────────────────────────────────────────────────────

def run_rag(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
) -> Tuple[str, List[Dict]]:
    """
    Synchronous RAG pipeline.

    Returns:
        answer      — LLM text with [#N] citation markers
        cited       — meetings referenced in the answer, in citation order
    """
    k = top_k or settings.rag_top_k
    query_vec = embed_query(question)
    candidates = vector_search(query_vec, top_k=k, filters=filters)
    answer = _generate(question, candidates)
    cited = _cited_meetings(answer, candidates)
    return _clean_answer(answer), cited


async def async_run_rag(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
) -> Tuple[str, List[Dict]]:
    """Async RAG pipeline for FastAPI."""
    import asyncio
    k = top_k or settings.rag_top_k
    loop = asyncio.get_event_loop()
    query_vec = await loop.run_in_executor(None, embed_query, question)
    candidates = await async_vector_search(query_vec, top_k=k, filters=filters)
    answer = await loop.run_in_executor(None, _generate, question, candidates)
    cited = _cited_meetings(answer, candidates)
    return _clean_answer(answer), cited


# ── Internal helpers ──────────────────────────────────────────────────────────

def _generate(question: str, candidates: List[Dict]) -> str:
    """Call Gemini with the candidate meetings as context."""
    context = _build_context(candidates)
    prompt = (
        f"--- RETRIEVED MEETINGS ---\n{context}\n--- END ---\n\n"
        f"User question: {question}\n\n"
        f"Answer, referencing each meeting you mention with [#N]:"
    )
    resp = _llm_client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            max_output_tokens=settings.rag_max_tokens,
            temperature=0.2,
        ),
    )
    return resp.text.strip()


def _cited_meetings(answer: str, candidates: List[Dict]) -> List[Dict]:
    """
    Parse [#N] markers from the answer and return the referenced meetings
    in the order they are first cited. Meetings not mentioned are excluded.
    Falls back to returning all candidates if no markers found.
    """
    indices = []
    seen = set()
    for m in re.finditer(r'\[#(\d+)\]', answer):
        n = int(m.group(1))
        if n not in seen and 1 <= n <= len(candidates):
            seen.add(n)
            indices.append(n - 1)   # convert 1-based to 0-based

    if not indices:
        # LLM didn't use markers — return all candidates
        return candidates

    return [candidates[i] for i in indices]


def _build_context(meetings: List[Dict]) -> str:
    """Build the numbered meeting list sent to the LLM."""
    if not meetings:
        return "No meetings found matching the search criteria."

    parts = []
    for i, m in enumerate(meetings, 1):
        lines = [f"[#{i}] {m.get('name', 'Unknown')}"]

        day  = m.get("day_text") or _day_num_to_name(m.get("day"))
        time = m.get("time_formatted") or m.get("time", "")
        if day or time:
            lines.append(f"  Day/Time: {day} {time}".strip())

        loc  = m.get("location", "")
        addr = m.get("formatted_address", "")
        if loc:
            lines.append(f"  Location: {loc}")
        if addr and addr != loc:
            lines.append(f"  Address:  {addr}")

        if m.get("region"):
            lines.append(f"  Region:   {m['region']}")

        types = m.get("types", [])
        if types:
            labels = m.get("type_labels") or types
            lines.append(f"  Types:    {', '.join(labels if isinstance(labels, list) else [labels])}")

        att = m.get("attendance_option", "")
        if att:
            lines.append(f"  Attendance: {att.replace('_', ' ')}")

        notes = (m.get("notes") or "").strip()
        if notes:
            lines.append(f"  Notes:    {notes[:200]}")

        if m.get("url"):
            lines.append(f"  URL:      {m['url']}")

        score = m.get("score")
        if score is not None:
            lines.append(f"  Score:    {score:.3f}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


_DAY_NAMES = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

def _day_num_to_name(day) -> str:
    try:
        return _DAY_NAMES[int(day)]
    except (TypeError, ValueError, IndexError):
        return str(day) if day is not None else ""
