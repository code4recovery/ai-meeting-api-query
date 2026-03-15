#!/usr/bin/env python3
"""
ingest_local.py — Load meetings.json → generate Gemini embeddings → upsert into MongoDB Atlas.

Tailored to the exact schema of the Harbor Area Central Office meetings export.

Usage:
    python ingest_local.py --file meetings.json
    python ingest_local.py --file meetings.json --skip-inactive
    python ingest_local.py --file meetings.json --limit 50 --skip-inactive
"""
# Add this to the top of ingest_local.py, before MongoClient is called
import pymongo
import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Deps ──────────────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    from pymongo import MongoClient, UpdateOne
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeElapsedColumn, MofNCompleteColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install google-genai pymongo python-dotenv rich tenacity")
    sys.exit(1)

load_dotenv(Path(__file__).resolve().parent / ".env")
console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL   = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
MONGODB_URI          = os.getenv("MONGODB_URI", "")
MONGODB_DB           = os.getenv("MONGODB_DB", "aa_meetings")
MONGODB_COLLECTION   = os.getenv("MONGODB_COLLECTION", "meetings")
VECTOR_INDEX_NAME    = os.getenv("VECTOR_INDEX_NAME", "aa_meetings_vector_index")
EMBEDDING_DIMENSIONS = 3072  # gemini-embedding-001 output size

# Day name lookup
DAY_NAMES = {0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",
             4:"Thursday",5:"Friday",6:"Saturday"}

# Human-readable type labels (full set from this dataset)
TYPE_LABELS = {
    "12x12":"12 Steps & 12 Traditions","1HR":"One Hour Meeting",
    "A":"Atheist/Agnostic","ABSI":"As Bill Sees It",
    "ASL":"American Sign Language","B":"Beginners",
    "BE":"Beginner/Newcomer","BS":"Book Study","C":"Closed",
    "CAN":"Candlelight","CF":"Child Friendly","D":"Discussion",
    "DR":"Daily Reflections","EN":"En Español","InP":"In Person",
    "LGBTQ":"LGBTQ+","LIT":"Literature","LS":"Living Sober",
    "M":"Men","MED":"Meditation","NB":"Non-Binary","O":"Open",
    "ONL":"Online/Virtual","Part":"Participation","QA":"Question & Answer",
    "S":"Spanish","SEN":"Seniors","SP":"Speaker","SS":"Step Study",
    "ST":"Step","T":"Traditions","TC":"Temporary Closed Contact",
    "VM":"Virtual Meeting","W":"Women","X":"Open to All",
    "XT":"Open to All (extended)","Y":"Young People",
}


# ── Embedding ─────────────────────────────────────────────────────────────────

_genai_client = None

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=15))
def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    client = get_genai_client()
    result = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values


def build_embed_text(m: dict) -> str:
    """
    Build a rich plain-text representation of a meeting for embedding.
    More descriptive text → better semantic search quality.
    """
    parts = []

    name = m.get("name","")
    if name:
        parts.append(f"Meeting name: {name}")

    location = m.get("location","")
    addr = m.get("formatted_address","")
    if location:
        parts.append(f"Location: {location}")
    if addr and addr != location:
        parts.append(f"Address: {addr}")

    region = m.get("region","")
    if region:
        parts.append(f"Region: {region}")

    day_num = m.get("day")
    day_text = DAY_NAMES.get(day_num, "") if day_num is not None else ""
    time_str = m.get("time_formatted","") or m.get("time","")
    end_time = m.get("end_time","")
    if day_text:
        parts.append(f"Day: {day_text}")
    if time_str:
        t = f"Time: {time_str}"
        if end_time:
            t += f" to {end_time}"
        parts.append(t)

    types = m.get("types",[])
    if types:
        labels = [TYPE_LABELS.get(t, t) for t in types]
        parts.append(f"Meeting types: {', '.join(labels)}")

    attendance = m.get("attendance_option","")
    if attendance:
        parts.append(f"Attendance: {attendance.replace('_',' ')}")

    notes = (m.get("notes","") or "").strip()
    if notes:
        parts.append(f"Notes: {notes[:300]}")

    conference = m.get("conference_url","")
    if conference:
        parts.append(f"Online meeting URL available")

    return ". ".join(parts)


# ── Normalise ─────────────────────────────────────────────────────────────────

def normalise(raw: dict) -> dict:
    """Map raw Meeting Guide record to our MongoDB schema."""
    day_num = raw.get("day")
    try:
        day_num = int(day_num)
    except (TypeError, ValueError):
        day_num = None

    types = raw.get("types", [])
    if isinstance(types, str):
        types = [t.strip() for t in types.split(",") if t.strip()]

    return {
        "meeting_id":        str(raw["id"]),
        "name":              raw.get("name",""),
        "slug":              raw.get("slug",""),
        "location":          raw.get("location",""),
        "location_url":      raw.get("location_url",""),
        "formatted_address": raw.get("formatted_address",""),
        "approximate":       raw.get("approximate",""),
        "latitude":          raw.get("latitude"),
        "longitude":         raw.get("longitude"),
        "region_id":         raw.get("region_id"),
        "region":            raw.get("region",""),
        "regions":           raw.get("regions",[]),
        "day":               day_num,
        "day_text":          DAY_NAMES.get(day_num,"") if day_num is not None else "",
        "time":              raw.get("time",""),
        "end_time":          raw.get("end_time",""),
        "time_formatted":    raw.get("time_formatted",""),
        "types":             types,
        "type_labels":       [TYPE_LABELS.get(t,t) for t in types],
        "attendance_option": raw.get("attendance_option",""),
        "notes":             raw.get("notes",""),
        "url":               raw.get("url",""),
        "conference_url":    raw.get("conference_url",""),
        "conference_phone":  raw.get("conference_phone",""),
        "entity":            raw.get("entity",""),
        "entity_url":        raw.get("entity_url",""),
        "updated":           raw.get("updated",""),
        "location_id":       raw.get("location_id"),
    }


# ── Atlas index definition ────────────────────────────────────────────────────

VECTOR_INDEX_DEF = {
    "name": VECTOR_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": EMBEDDING_DIMENSIONS,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "day"},
            {"type": "filter", "path": "region"},
            {"type": "filter", "path": "types"},
            {"type": "filter", "path": "attendance_option"},
        ]
    },
}


# ── Main ingest ───────────────────────────────────────────────────────────────


def _resolve_srv(uri: str) -> str:
    """Resolve mongodb+srv:// using Google DNS to bypass broken system resolvers."""
    if not uri.startswith("mongodb+srv://"):
        return uri
    try:
        import dns.resolver
        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
        after = uri[len("mongodb+srv://"):]
        creds, rest = (after.split("@", 1) if "@" in after else (None, after))
        hostname = rest.split("/")[0].split("?")[0]
        answers = resolver.resolve(f"_mongodb._tcp.{hostname}", "SRV")
        hosts = ",".join(f"{r.target.to_text().rstrip('.')}:{r.port}" for r in answers)
        db_and_opts = rest[len(hostname):]
        cred_prefix = f"{creds}@" if creds else ""
        standard = f"mongodb://{cred_prefix}{hosts}{db_and_opts}"
        sep = "&" if "?" in standard else "?"
        if "ssl" not in standard and "tls" not in standard:
            standard += f"{sep}ssl=true"; sep = "&"
        if "authSource" not in standard:
            standard += f"{sep}authSource=admin"
        return standard
    except Exception:
        return uri


def run(file_path: str, skip_inactive: bool = False, limit: int = 0, dry_run: bool = False):
    console.rule("[bold blue]AA Meetings — Gemini + MongoDB Atlas Ingestion[/bold blue]")

    # ── Validate config ───────────────────────────────────────────────────
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not MONGODB_URI:
        missing.append("MONGODB_URI")
    if missing:
        console.print(f"[red]Missing env vars: {', '.join(missing)}[/red]")
        console.print("Create a .env file — see .env.example")
        sys.exit(1)

    get_genai_client()  # initialise client
    console.print(f"[green]✓[/green] Gemini configured  (embedding model: {GEMINI_EMBED_MODEL})")

    # ── Load JSON ─────────────────────────────────────────────────────────
    p = Path(file_path)
    if not p.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        sys.exit(1)

    with open(p) as f:
        raw_data = json.load(f)

    console.print(f"[green]✓[/green] Loaded {len(raw_data)} records from [bold]{p.name}[/bold]")

    # ── Filter ────────────────────────────────────────────────────────────
    if skip_inactive:
        before = len(raw_data)
        raw_data = [m for m in raw_data if m.get("attendance_option") != "inactive"]
        console.print(f"  Skipped {before - len(raw_data)} inactive meetings → {len(raw_data)} remaining")

    if limit and limit > 0:
        raw_data = raw_data[:limit]
        console.print(f"  Limit applied → {len(raw_data)} meetings to process")

    if not raw_data:
        console.print("[yellow]No meetings to process.[/yellow]")
        return

    # ── Normalise ─────────────────────────────────────────────────────────
    meetings = [normalise(m) for m in raw_data]

    # ── Print sample ──────────────────────────────────────────────────────
    console.print("\n[bold]Sample record (normalised):[/bold]")
    sample = {k: v for k, v in meetings[0].items() if k not in ("embedding",)}
    console.print(sample)
    console.print(f"\n[bold]Sample embed text:[/bold]\n{build_embed_text(meetings[0])}\n")

    if dry_run:
        console.print("[yellow]--dry-run: stopping before embeddings + DB write[/yellow]")
        return

    # ── Generate embeddings ───────────────────────────────────────────────
    console.print(f"\n[cyan]Generating embeddings for {len(meetings)} meetings...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(meetings))
        for i, m in enumerate(meetings):
            doc_text = build_embed_text(m)
            m["embedding"]      = embed_text(doc_text, task_type="retrieval_document")
            m["embedding_text"] = doc_text   # stored for transparency/debugging
            progress.advance(task)
            # ~1500 RPM free-tier limit — brief pause every 100 calls
            if (i + 1) % 100 == 0:
                time.sleep(1)

    console.print(f"[green]✓[/green] Embeddings complete  (dim={len(meetings[0]['embedding'])})")

    # ── Upsert to MongoDB ─────────────────────────────────────────────────
    console.print(f"\n[cyan]Connecting to MongoDB Atlas...[/cyan]")
    resolved_uri = _resolve_srv(MONGODB_URI)
    client = MongoClient(resolved_uri, serverSelectionTimeoutMS=10_000)
    try:
        client.admin.command("ping")
    except Exception as e:
        console.print(f"[red]MongoDB connection failed: {e}[/red]")
        sys.exit(1)
    console.print(f"[green]✓[/green] Connected  →  db=[bold]{MONGODB_DB}[/bold]  collection=[bold]{MONGODB_COLLECTION}[/bold]")

    col = client[MONGODB_DB][MONGODB_COLLECTION]

    ops = [
        UpdateOne({"meeting_id": m["meeting_id"]}, {"$set": m}, upsert=True)
        for m in meetings
    ]
    result = col.bulk_write(ops, ordered=False)
    console.print(
        f"[green]✓[/green] Upsert complete — "
        f"inserted: [bold]{result.upserted_count}[/bold]  "
        f"modified: [bold]{result.modified_count}[/bold]"
    )

    # ── Regular indexes ───────────────────────────────────────────────────
    col.create_index("meeting_id", unique=True, background=True)
    col.create_index("day",               background=True)
    col.create_index("region",            background=True)
    col.create_index("types",             background=True)
    col.create_index("attendance_option", background=True)
    console.print("[green]✓[/green] Regular indexes ensured")

    # ── Print Atlas Vector Search Index definition ────────────────────────
    console.rule("[bold yellow]Next Step — Create Atlas Vector Search Index[/bold yellow]")
    console.print(
        "\nIn [bold]MongoDB Atlas UI[/bold]:\n"
        "  1. Open your cluster → Browse Collections → [bold]aa_meetings.meetings[/bold]\n"
        "  2. Go to [bold]Atlas Search[/bold] tab → [bold]Create Search Index[/bold]\n"
        "  3. Choose [bold]JSON Editor[/bold] and paste the definition below:\n"
    )
    console.print(Panel(
        json.dumps(VECTOR_INDEX_DEF, indent=2),
        title="[bold cyan]Vector Search Index Definition[/bold cyan]",
        border_style="cyan",
    ))

    # ── Summary table ─────────────────────────────────────────────────────
    table = Table(title="Ingestion Summary", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total meetings processed", str(len(meetings)))
    table.add_row("Inserted (new)",            str(result.upserted_count))
    table.add_row("Modified (updated)",        str(result.modified_count))
    table.add_row("Embedding model",           GEMINI_EMBED_MODEL)
    table.add_row("Embedding dimensions",      str(EMBEDDING_DIMENSIONS))
    table.add_row("MongoDB database",          MONGODB_DB)
    table.add_row("MongoDB collection",        MONGODB_COLLECTION)
    table.add_row("Vector index name",         VECTOR_INDEX_NAME)

    # Attendance breakdown
    from collections import Counter
    att_counts = Counter(m.get("attendance_option","") for m in meetings)
    for att, cnt in sorted(att_counts.items()):
        table.add_row(f"  · {att}", str(cnt))

    console.print(table)
    console.rule("[bold green]Done! Ready to query.[/bold green]")
    console.print(
        "\nOnce the Atlas vector index is active, run:\n"
        "  [bold]python src/cli.py[/bold]         — interactive chat\n"
        "  [bold]uvicorn src.api:app --reload[/bold]  — REST API\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest local meetings.json into MongoDB + Gemini embeddings")
    parser.add_argument("--file",          default="meetings.json", help="Path to meetings.json")
    parser.add_argument("--skip-inactive", action="store_true",      help="Skip inactive meetings (170 records)")
    parser.add_argument("--limit",         type=int, default=0,      help="Max records to process (0=all)")
    parser.add_argument("--dry-run",       action="store_true",      help="Normalise + preview without writing to DB")
    args = parser.parse_args()
    run(
        file_path=args.file,
        skip_inactive=args.skip_inactive,
        limit=args.limit,
        dry_run=args.dry_run,
    )
