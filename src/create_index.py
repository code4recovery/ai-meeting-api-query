#!/usr/bin/env python3
"""
create_index.py — Create the Atlas Vector Search index programmatically.
Standalone, no src imports needed.

Usage:
    python create_index.py
"""
import sys, os, json, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

env = {}
for line in open(PROJECT_ROOT / ".env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")

def cfg(k, d=""): return env.get(k) or os.environ.get(k, d)

MONGODB_URI  = cfg("MONGODB_URI")
MONGODB_DB   = cfg("MONGODB_DB",   "aa_meetings")
MONGODB_COL  = cfg("MONGODB_COLLECTION", "meetings")
INDEX_NAME   = cfg("VECTOR_INDEX_NAME",  "aa_meetings_vector_index")
EMBED_DIMS   = int(cfg("EMBEDDING_DIMENSIONS", "3072"))

from pymongo import MongoClient
from rich.console import Console
console = Console()

console.rule("[bold blue]Create Atlas Vector Search Index[/bold blue]")

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
db  = client[MONGODB_DB]
col = db[MONGODB_COL]

# ── Verify docs + dims ────────────────────────────────────────────────────────
total = col.count_documents({})
console.print(f"Collection: {MONGODB_DB}.{MONGODB_COL}  ({total} docs)")

doc = col.find_one({"embedding": {"$exists": True}}, {"embedding": 1})
if not doc:
    console.print("[red]No documents with embeddings found — run ingest_local.py first[/red]")
    sys.exit(1)

actual_dims = len(doc["embedding"])
console.print(f"Stored embedding dims: {actual_dims}")
if actual_dims != EMBED_DIMS:
    console.print(f"[yellow]Note: EMBEDDING_DIMENSIONS in .env is {EMBED_DIMS}, "
                  f"but stored docs have {actual_dims}. Using {actual_dims}.[/yellow]")
    EMBED_DIMS = actual_dims

# ── Check if index already exists ────────────────────────────────────────────
existing = [i.get("name") for i in col.list_search_indexes()]
if INDEX_NAME in existing:
    console.print(f"[green]Index {INDEX_NAME!r} already exists.[/green]")
    for idx in col.list_search_indexes():
        if idx.get("name") == INDEX_NAME:
            console.print(f"  Status: {idx.get('status')}")
    sys.exit(0)

# ── Create the index ──────────────────────────────────────────────────────────
index_def = {
    "name": INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type":          "vector",
                "path":          "embedding",
                "numDimensions": EMBED_DIMS,
                "similarity":    "cosine",
            },
            {"type": "filter", "path": "day"},
            {"type": "filter", "path": "region"},
            {"type": "filter", "path": "types"},
            {"type": "filter", "path": "attendance_option"},
        ]
    }
}

console.print(f"\nCreating index [bold]{INDEX_NAME!r}[/bold] with {EMBED_DIMS} dims...")
console.print("Definition:")
console.print(json.dumps(index_def["definition"], indent=2))

try:
    col.create_search_index(index_def)
    console.print("\n[green]✓ Index creation request accepted.[/green]")
except Exception as e:
    console.print(f"\n[red]create_search_index() failed: {e}[/red]")
    console.print("\n[yellow]Falling back: paste this JSON into Atlas UI manually:[/yellow]")
    console.print("[bold]Atlas → your cluster → Search Indexes → Create Search Index → JSON Editor[/bold]")
    console.print(json.dumps(index_def, indent=2))
    sys.exit(1)

# ── Poll until READY ──────────────────────────────────────────────────────────
console.print("\nWaiting for index to become READY (this takes 1–3 minutes)...")
for attempt in range(24):   # up to ~2 min
    time.sleep(5)
    try:
        indexes = list(col.list_search_indexes())
        for idx in indexes:
            if idx.get("name") == INDEX_NAME:
                status = idx.get("status", "UNKNOWN")
                console.print(f"  [{attempt*5}s] status = {status}")
                if status == "READY":
                    console.print(f"\n[green bold]✓ Index is READY — run cli.py or trace_query.py[/green bold]")
                    sys.exit(0)
                elif status == "FAILED":
                    console.print(f"[red]Index creation FAILED. Check Atlas UI for details.[/red]")
                    sys.exit(1)
    except Exception as e:
        console.print(f"  [{attempt*5}s] polling error: {e}")

console.print("\n[yellow]Index still building — check Atlas UI in a few minutes.[/yellow]")
console.print("Run  python trace_query.py  once it shows READY.")
