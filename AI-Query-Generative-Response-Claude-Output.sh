"""
RAG: Claude + AA Meetings (local JSON) + In-Memory VectorStore
====================================================================

Answers natural-language questions about AA meetings across the United States
by combining a knowledge base of AA terminology with structured meeting data
loaded from a local meetings.json file.

Each meeting record in meetings.json contains fields such as:
  - name          : meeting name
  - day           : day of week as integer (0=Sunday … 6=Saturday)
  - time / end_time / time_formatted : scheduled time
  - attendance_option : "in_person", "online", "hybrid", or "inactive"
  - types         : list of format/demographic codes (e.g. SP, W, BE, ONL)
  - formatted_address / region : location details
  - conference_url / conference_phone : Zoom or dial-in info for online meetings
  - notes         : free-text notes including passwords or access instructions
  - timezone      : IANA timezone string (e.g. "America/New_York")
  - entity / entity_url : the AA central office or intergroup that owns the record

Architecture:
  1. InMemoryVectorStore — zero-dependency TF-IDF cosine-similarity retrieval
                           used to surface relevant AA terminology context
  2. Data Source         — reads meetings.json from the same directory as this
                           script; optional live REST endpoint is commented out
  3. RAG Pipeline        — retrieves terminology context, filters meeting records
                           by intent (day, city/state, attendance, type), and
                           passes both to Claude to produce a grounded answer

Usage:
  pip install anthropic requests
  export ANTHROPIC_API_KEY=sk-...
  python3 rag.py

Data file:
  Place meetings.json in the same directory as this script.
  The file must be a JSON array of meeting objects. Many AA intergroup and
  central office websites expose this format via a WordPress plugin endpoint:
  https://<your-aa-site>/wp-admin/admin-ajax.php?action=meetings
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic
import requests

# ---------------------------------------------------------------------------
# 1. IN-MEMORY VECTOR STORE
# ---------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    text: str
    metadata: dict
    vector: dict[str, float]


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict
    score: float


class InMemoryVectorStore:
    """Zero-dependency vector store using TF-IDF + cosine similarity."""

    def __init__(self) -> None:
        self._docs: list[Document] = []

    # -- private helpers --------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if t]

    @staticmethod
    def _tf(tokens: list[str]) -> dict[str, float]:
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = len(tokens) or 1
        return {k: v / total for k, v in freq.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        dot = norm_a = norm_b = 0.0
        for k in set(a) | set(b):
            av, bv = a.get(k, 0.0), b.get(k, 0.0)
            dot   += av * bv
            norm_a += av * av
            norm_b += bv * bv
        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        return dot / denom if denom else 0.0

    # -- public API --------------------------------------------------------

    def add(self, id: str, text: str, metadata: dict | None = None) -> "InMemoryVectorStore":
        tokens = self._tokenize(text)
        self._docs.append(Document(id=id, text=text, metadata=metadata or {}, vector=self._tf(tokens)))
        return self

    def query(self, query_text: str, top_k: int = 3) -> list[SearchResult]:
        q_vec = self._tf(self._tokenize(query_text))
        scored = [
            SearchResult(id=d.id, text=d.text, metadata=d.metadata,
                         score=self._cosine(q_vec, d.vector))
            for d in self._docs
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 2. KNOWLEDGE BASE — AA meeting terminology & context
# ---------------------------------------------------------------------------

KNOWLEDGE_DOCS = [
    {
        "id": "meeting-types-format",
        "text": (
            "AA meeting formats include: Speaker (SP) — one member shares their personal story "
            "of alcoholism and recovery; Discussion (D) — members share on a chosen topic; "
            "Big Book Study (B/BS) — reading and discussing the AA Big Book chapter by chapter; "
            "Step Study (ST/SS) — working through the 12 Steps; 12x12 — studying the Twelve "
            "Steps and Twelve Traditions book; Participation (Part) — open sharing encouraged "
            "from all attendees; Literature (LIT) — study of AA-approved literature; "
            "Candlelight (CF) — meeting held by candlelight, often more intimate in tone."
        ),
        "metadata": {"topic": "meeting-types"},
    },
    {
        "id": "meeting-types-special",
        "text": (
            "AA meeting demographic and access designations found in the types field: "
            "Open (O) — anyone may attend, not just people with a drinking problem; "
            "Closed (C) — for those who identify as alcoholic or have a desire to stop drinking; "
            "Men (M) — men only; Women (W) — women only; "
            "LGBTQ — affirming meeting for LGBTQ+ members; "
            "Spanish (S) — conducted in Spanish; "
            "Beginners (BE) — specifically welcoming newcomers and people in early recovery; "
            "Seniors (SEN) — geared toward older members; "
            "Young People (Y) — for younger members, typically teens and young adults; "
            "Secular (A) — non-religious, spirituality-optional approach to the steps; "
            "Deaf/Hard of Hearing (deaf) — ASL interpreted or deaf-led meetings."
        ),
        "metadata": {"topic": "meeting-types-special"},
    },
    {
        "id": "attendance-options",
        "text": (
            "Every meeting record has an attendance_option field with one of four values: "
            "in_person — meets at a physical address only; "
            "online — virtual only, accessible via a Zoom link or phone dial-in; "
            "hybrid — simultaneous in-person and online attendance both supported; "
            "inactive — meeting is no longer running and should be ignored. "
            "Online and hybrid meetings include a conference_url (Zoom or Google Meet link) "
            "and often a conference_phone dial-in number. Access passwords or meeting IDs "
            "are usually in the notes field."
        ),
        "metadata": {"topic": "attendance"},
    },
    {
        "id": "meeting-days-times",
        "text": (
            "The day field is an integer: 0=Sunday, 1=Monday, 2=Tuesday, 3=Wednesday, "
            "4=Thursday, 5=Friday, 6=Saturday. "
            "The time field uses 24-hour HH:MM format; time_formatted shows 12-hour am/pm. "
            "end_time gives the scheduled end in HH:MM. Most meetings run 60 or 90 minutes. "
            "The timezone field contains an IANA zone string (e.g. America/New_York, "
            "America/Chicago, America/Denver, America/Los_Angeles, America/Phoenix) "
            "which is essential for cross-timezone queries."
        ),
        "metadata": {"topic": "schedule"},
    },
    {
        "id": "location-fields",
        "text": (
            "Location data in each meeting record: formatted_address is the full street "
            "address including city, state, and ZIP code — this is the primary field for "
            "geographic filtering across the United States. The region field holds the "
            "local area or district name used by the sponsoring intergroup. "
            "latitude and longitude enable distance-based lookups. "
            "When approximate is 'yes' the address is a city or ZIP centroid, not a "
            "precise venue. location holds the venue name (e.g. 'Community Church', "
            "the location_notes field has room numbers or entry directions."
        ),
        "metadata": {"topic": "location"},
    },
    {
        "id": "online-access",
        "text": (
            "For online and hybrid meetings the conference_url field contains a full Zoom "
            "or Google Meet link. The conference_phone field lists a dial-in number for "
            "members without internet access. Meeting passwords, passcodes, or numeric IDs "
            "needed to join are in the notes field, often prefixed with 'Password:' or "
            "'UPDATE: Password:'. Common Zoom regional dial-in numbers include "
            "646-558-8656 (New York), 312-626-6799 (Chicago), 669-900-6833 (California), "
            "253-215-8782 (Pacific Northwest), and 301-715-8592 (Mid-Atlantic)."
        ),
        "metadata": {"topic": "zoom-online"},
    },
    {
        "id": "entity-intergroup",
        "text": (
            "The entity field identifies the AA intergroup, central office, or area that "
            "maintains the meeting record. Intergroups are local service bodies that support "
            "AA groups across a city, county, or multi-county area. The entity_url field "
            "links to the intergroup's website. A single meetings.json file may contain "
            "records from one intergroup or be aggregated from multiple intergroups across "
            "different cities and states."
        ),
        "metadata": {"topic": "entity"},
    },
    {
        "id": "aa-overview",
        "text": (
            "Alcoholics Anonymous (AA) is a worldwide fellowship of people who share their "
            "experience, strength, and hope to solve their common problem and help others "
            "recover from alcoholism. AA has no dues or fees and is fully self-supporting. "
            "The only requirement for membership is a desire to stop drinking. "
            "AA meetings are held in all 50 US states and in over 180 countries. "
            "The program is built on 12 Steps and guided by 12 Traditions. "
            "Anonymity is a foundational principle — last names and personal details "
            "are kept confidential within the fellowship."
        ),
        "metadata": {"topic": "aa-overview"},
    },
]


def build_knowledge_base() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    for doc in KNOWLEDGE_DOCS:
        store.add(doc["id"], doc["text"], doc["metadata"])
    print(f"✅ Knowledge base loaded ({len(KNOWLEDGE_DOCS)} documents)\n")
    return store


# ---------------------------------------------------------------------------
# 3. DATA SOURCE — local JSON file (with optional REST API endpoint)
# ---------------------------------------------------------------------------

# Path to the local meetings data file (same directory as this script)
MEETINGS_JSON_PATH = Path(__file__).parent / "meetings.json"

# ---------------------------------------------------------------------------
# Optional: to fetch live data directly from an AA intergroup REST endpoint
# instead of the local file, comment out the local-file block in
# load_meetings_data() below and uncomment the requests block.
# Replace the URL with your intergroup's endpoint — many AA sites expose
# meeting data via the Meeting Guide WordPress plugin at:
#   https://<intergroup-site>/wp-admin/admin-ajax.php?action=meetings
# No API key is required for public endpoints.
#
# MEETINGS_API_URL = "https://hacoaa.org/wp-admin/admin-ajax.php?action=meetings"
# ---------------------------------------------------------------------------

DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def load_meetings_data() -> list[dict]:
    """
    Load raw meeting records from the configured data source.

    LOCAL FILE (default):
      Reads meetings.json from the same directory as this script.
      The file must be a JSON array of meeting objects — the standard format
      used by the Meeting Guide WordPress plugin deployed on many AA intergroup
      and central office websites across the United States.

    REST API (optional):
      Comment out the local-file block below and uncomment the requests block
      to pull live data directly from an intergroup endpoint instead.
    """
    # -- Local file (default) ------------------------------------------------
    if not MEETINGS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"meetings.json not found at {MEETINGS_JSON_PATH}\n"
            "Export it from your AA intergroup site or download a sample with:\n"
            "  curl -o meetings.json "
            "'https://<your-intergroup-site>/wp-admin/admin-ajax.php?action=meetings'"
        )
    with MEETINGS_JSON_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)

    # -- REST API (optional — uncomment to use) ------------------------------
    # resp = requests.get(MEETINGS_API_URL, timeout=15)
    # resp.raise_for_status()
    # return resp.json()


def fetch_meetings(
    *,
    day: int | None = None,
    region: str | None = None,
    attendance_option: str | None = None,
    search_term: str | None = None,
) -> dict:
    """
    Load meetings from meetings.json and apply optional filters.

    Parameters
    ----------
    day              : int 0–6 (0=Sunday) — filter to a specific day of the week
    region           : str — matched against region, formatted_address, and
                       timezone fields so callers can pass a city, state name,
                       state abbreviation, or ZIP code prefix
    attendance_option: "in_person" | "online" | "hybrid" — filter by format
    search_term      : str — matched against meeting name, type codes, location
                       name, and notes (useful for "women", "beginner", "spanish")
    """
    try:
        all_meetings: list[dict] = load_meetings_data()
    except (FileNotFoundError, json.JSONDecodeError, requests.RequestException) as exc:
        return {"error": str(exc)}

    # Remove inactive meetings
    active = [m for m in all_meetings if m.get("attendance_option") != "inactive"]
    active_count = len(active)

    meetings = active

    # Filter by day of week
    if day is not None:
        meetings = [m for m in meetings if m.get("day") == day]

    # Filter by location — check region name, full address, and timezone.
    # Also expand US state full names to their 2-letter abbreviations so that
    # a query for "Texas" matches addresses containing ", TX".
    if region:
        STATE_ABBREVS = {
            "alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca",
            "colorado":"co","connecticut":"ct","delaware":"de","florida":"fl","georgia":"ga",
            "hawaii":"hi","idaho":"id","illinois":"il","indiana":"in","iowa":"ia",
            "kansas":"ks","kentucky":"ky","louisiana":"la","maine":"me","maryland":"md",
            "massachusetts":"ma","michigan":"mi","minnesota":"mn","mississippi":"ms",
            "missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv",
            "new hampshire":"nh","new jersey":"nj","new mexico":"nm","new york":"ny",
            "north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok",
            "oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc",
            "south dakota":"sd","tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt",
            "virginia":"va","washington":"wa","west virginia":"wv","wisconsin":"wi",
            "wyoming":"wy",
        }
        r_lower = region.lower()
        # Expand full state name to abbreviation (e.g. "texas" → "tx")
        r_abbrev = STATE_ABBREVS.get(r_lower)
        def _loc_match(m: dict) -> bool:
            addr = (m.get("formatted_address") or "").lower()
            reg  = (m.get("region") or "").lower()
            tz   = (m.get("timezone") or "").lower()
            if r_lower in addr or r_lower in reg or r_lower in tz:
                return True
            if r_abbrev and f", {r_abbrev}" in addr:
                return True
            return False
        meetings = [m for m in meetings if _loc_match(m)]

    if attendance_option:
        meetings = [m for m in meetings if m.get("attendance_option") == attendance_option]

    if search_term:
        s = search_term.lower()
        def _match(m: dict) -> bool:
            return (
                s in (m.get("name") or "").lower()
                or any(s in t.lower() for t in (m.get("types") or []))
                or s in (m.get("location") or "").lower()
                or s in (m.get("notes") or "").lower()
            )
        meetings = [m for m in meetings if _match(m)]

    # Shape into clean summaries for the prompt
    shaped = [
        {
            "id":         m["id"],
            "name":       m.get("name"),
            "day":        DAY_NAMES[m["day"]] if 0 <= m.get("day", -1) <= 6 else f"Day {m.get('day')}",
            "time":       m.get("time_formatted"),
            "end_time":   m.get("end_time"),
            "timezone":   m.get("timezone"),
            "attendance": m.get("attendance_option"),
            "types":      m.get("types") or [],
            "location":   m.get("location"),
            "address":    m.get("region") if m.get("approximate") == "yes" else m.get("formatted_address"),
            "region":     m.get("region"),
            "entity":     m.get("entity"),
            "zoom_url":   m.get("conference_url"),
            "zoom_phone": m.get("conference_phone"),
            "notes":      m.get("notes"),
            "url":        m.get("url"),
        }
        for m in meetings
    ]

    return {
        "total_in_db":     len(all_meetings),
        "active_meetings": active_count,
        "filters_applied": {
            "day": day,
            "region": region,
            "attendance_option": attendance_option,
            "search_term": search_term,
        },
        "count":    len(shaped),
        "meetings": shaped,
        "source":   "meetings.json",
    }


# ---------------------------------------------------------------------------
# 4. QUERY INTENT PARSER
# ---------------------------------------------------------------------------

# City and state name fragments matched against formatted_address, region,
# and timezone fields. Extend this list freely — any substring that uniquely
# identifies a location in your meetings.json will work.
KNOWN_REGIONS = [
    # US states (full names and common abbreviations)
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming",
    # Major US cities
    "new york city", "los angeles", "chicago", "houston", "phoenix",
    "philadelphia", "san antonio", "san diego", "dallas", "san jose",
    "austin", "jacksonville", "san francisco", "columbus", "charlotte",
    "indianapolis", "seattle", "denver", "nashville", "boston",
    "portland", "las vegas", "memphis", "louisville", "baltimore",
    "milwaukee", "albuquerque", "tucson", "fresno", "sacramento",
    "kansas city", "mesa", "atlanta", "omaha", "colorado springs",
    "raleigh", "minneapolis", "cleveland", "wichita", "new orleans",
    "arlington", "bakersfield", "tampa", "aurora", "honolulu",
    "anaheim", "santa ana", "corpus christi", "riverside", "lexington",
    "st. louis", "pittsburgh", "anchorage", "stockton", "cincinnati",
    "st. paul", "toledo", "greensboro", "newark", "plano",
    "henderson", "lincoln", "orlando", "jersey city", "chandler",
    "st. petersburg", "laredo", "norfolk", "madison", "durham",
    "lubbock", "winston-salem", "garland", "glendale", "hialeah",
    "reno", "baton rouge", "irvine", "chesapeake", "scottsdale",
    "north las vegas", "fremont", "gilbert", "san bernardino", "birmingham",
    "rochester", "richmond", "spokane", "des moines", "montgomery",
    "modesto", "fayetteville", "tacoma", "shreveport", "salt lake city",
    "little rock", "oxnard", "providence", "knoxville", "grand rapids",
    "bridgeport", "fort wayne", "huntsville", "worcester", "brownsville",
    "tempe", "santa clarita", "garden grove", "oceanside", "fort lauderdale",
    "long beach", "santa ana", "elk grove", "clarksville", "rockford",
]

TYPE_KEYWORDS = [
    "women", "mens", "spanish", "beginner", "speaker",
    "big book", "step study", "discussion", "lgbtq", "secular",
    "candlelight", "literature", "young people", "seniors", "deaf",
]


def parse_intent(question: str) -> dict:
    """
    Extract structured filter parameters from a natural-language question.

    Detects:
      day              — day-of-week keywords ("sunday", "monday", …, "today", "tomorrow")
      region           — US city or state name matched against KNOWN_REGIONS;
                         passed to fetch_meetings which checks formatted_address,
                         region, and timezone fields
      attendance_option — "online" / "in_person" / "hybrid" from keyword cues
      search_term      — meeting type or demographic keyword (women, beginner, etc.)
    """
    q = question.lower()
    today_dow = datetime.now().weekday()          # Mon=0 … Sun=6 (Python)
    # Convert Python weekday (Mon=0) → JS-style (Sun=0)
    py_to_aa = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
    aa_today    = py_to_aa[today_dow]
    aa_tomorrow = py_to_aa[(today_dow + 1) % 7]

    day_map = {
        "sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
        "thursday": 4, "friday": 5, "saturday": 6,
        "tonight": aa_today, "today": aa_today, "tomorrow": aa_tomorrow,
    }
    day = next((num for word, num in day_map.items() if word in q), None)
    region = next((r for r in KNOWN_REGIONS if r in q), None)

    attendance_option = None
    if any(k in q for k in ("online", "zoom", "virtual")):
        attendance_option = "online"
    elif any(k in q for k in ("in person", "in-person")):
        attendance_option = "in_person"
    elif "hybrid" in q:
        attendance_option = "hybrid"

    search_term = next((t for t in TYPE_KEYWORDS if t in q), None)

    return {
        "day": day,
        "region": region,
        "attendance_option": attendance_option,
        "search_term": search_term,
    }


# ---------------------------------------------------------------------------
# 5. RAG PIPELINE
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful AA meeting finder assistant. The meeting data comes from a meetings.json
file that may contain AA meetings from anywhere in the United States.

Answer questions using:
1. The knowledge-base context below, which explains AA terminology, meeting type codes,
   attendance options, and how the meeting record fields are structured.
2. The meeting records provided, filtered from meetings.json based on the user's question.

When listing meetings be specific: include the meeting name, day, time (with timezone if
available), full address or city/state, attendance option, and — for online or hybrid
meetings — the Zoom link and any password from the notes field.
If no meetings match the filters, say so clearly and suggest broadening the search.
Keep answers friendly, concise, and easy to scan.\
"""


class RAGPipeline:
    def __init__(self, vector_store: InMemoryVectorStore, client: anthropic.Anthropic) -> None:
        self.store  = vector_store
        self.client = client

    def answer(self, question: str) -> dict:
        print("━" * 60)
        print(f"❓ Question: {question}")
        print("━" * 60)

        # Step 1 — vector store retrieval
        retrieved = self.store.query(question, top_k=3)
        print(f"\n📚 Retrieved {len(retrieved)} context chunk(s):")
        for r in retrieved:
            print(f"   [{r.id}] score={r.score:.3f}  \"{r.text[:70]}…\"")

        # Step 2 — load meeting data and apply intent filters
        intent = parse_intent(question)
        print(f"\n🔍 Parsed intent: {intent}")
        print("\n📂 Loading meetings from meetings.json…")

        meeting_data = fetch_meetings(**intent)
        if "error" in meeting_data:
            print(f"   ⚠️  Error: {meeting_data['error']}")
        else:
            print(f"   ✅ {meeting_data['count']} matching meetings "
                  f"({meeting_data['active_meetings']} active total)")

        # Trim to first 20 results to keep prompt size manageable
        prompt_data = meeting_data.copy()
        if "meetings" in prompt_data:
            prompt_data["meetings"] = prompt_data["meetings"][:20]

        # Step 3 — build prompt
        context_block = "\n\n".join(
            f"[{r.id}]\n{r.text.strip()}" for r in retrieved
        )

        if "error" in meeting_data:
            meetings_block = f"\n\n## Meetings Data\nError: {meeting_data['error']}"
        else:
            meetings_block = (
                "\n\n## Meetings Data (local meetings.json)\n"
                f"```json\n{json.dumps(prompt_data, indent=2)}\n```"
            )

        user_message = (
            f"## Retrieved Knowledge-Base Context\n{context_block}"
            + meetings_block
            + f"\n\n## Question\n{question}"
        )

        # Step 4 — Claude
        print("\n🤖 Calling Claude…")
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        print("\n💬 Claude's Answer:\n")
        print(answer_text)
        print()

        return {
            "question":     question,
            "intent":       intent,
            "retrieved":    [{"id": r.id, "score": r.score} for r in retrieved],
            "meeting_data": meeting_data,
            "answer":       answer_text,
            "usage":        {"input_tokens": response.usage.input_tokens,
                             "output_tokens": response.usage.output_tokens},
        }


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "Are there any online AA meetings on Sunday morning I can join via Zoom?",
    "What women's meetings are available in Chicago on weekday evenings?",
    "I'm new to AA — are there any beginner or newcomer meetings in Texas?",
    "Show me in-person speaker meetings on Saturday night.",
    "Are there any Spanish-language AA meetings available online?",
]


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌  ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client   = anthropic.Anthropic(api_key=api_key)
    store    = build_knowledge_base()
    pipeline = RAGPipeline(store, client)

    for question in EXAMPLE_QUESTIONS:
        pipeline.answer(question)


if __name__ == "__main__":
    main()

