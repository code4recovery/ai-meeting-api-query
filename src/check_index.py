#!/usr/bin/env python3
"""
check_index.py — Print the exact Atlas search index definition and test with a zero vector.
Standalone, no src imports.
"""
import sys, os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

env = {}
for line in open(PROJECT_ROOT / ".env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")

def cfg(k, d=""): return env.get(k) or os.environ.get(k, d)

MONGODB_URI   = cfg("MONGODB_URI")
MONGODB_DB    = cfg("MONGODB_DB", "aa_meetings")
MONGODB_COL   = cfg("MONGODB_COLLECTION", "meetings")
VECTOR_INDEX  = cfg("VECTOR_INDEX_NAME", "aa_meetings_vector_index")

from pymongo import MongoClient
from rich.console import Console
from rich.syntax import Syntax
console = Console()

console.rule("[bold blue]Atlas Index Inspector[/bold blue]")

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
col = client[MONGODB_DB][MONGODB_COL]

# ── 1. Print ALL search indexes verbatim ──────────────────────────────────────
console.print("\n[cyan]All search indexes on this collection:[/cyan]")
indexes = list(col.list_search_indexes())
if not indexes:
    console.print("[red]No search indexes found at all.[/red]")
    sys.exit(1)

for idx in indexes:
    name   = idx.get("name")
    status = idx.get("status")
    itype  = idx.get("type")
    console.print(f"\n  name={name!r}  status={status}  type={itype}")
    defn = idx.get("latestDefinition") or idx.get("definition") or {}
    console.print(Syntax(json.dumps(defn, indent=2), "json", theme="monokai"))

# ── 2. Check field path in stored docs ───────────────────────────────────────
console.print("\n[cyan]Checking embedding field in stored documents:[/cyan]")
doc = col.find_one({"embedding": {"$exists": True}})
if doc:
    emb = doc["embedding"]
    console.print(f"  Field name: 'embedding'  ✓")
    console.print(f"  Dims: {len(emb)}")
    console.print(f"  Type of first element: {type(emb[0]).__name__}")
    console.print(f"  First 3 values: {emb[:3]}")
else:
    console.print("[red]  No document with 'embedding' field found![/red]")
    sys.exit(1)

# ── 3. Try $vectorSearch with the exact index name and zero vector ────────────
console.print(f"\n[cyan]Testing $vectorSearch with index={VECTOR_INDEX!r}:[/cyan]")
zero = [0.0] * len(emb)
try:
    r = list(col.aggregate([
        {"$vectorSearch": {
            "index": VECTOR_INDEX,
            "path": "embedding",
            "queryVector": zero,
            "numCandidates": 10,
            "limit": 3,
        }},
        {"$project": {"name": 1, "day": 1}},
    ]))
    console.print(f"  Results: {len(r)}")
    for x in r:
        console.print(f"    {x}")
except Exception as e:
    console.print(f"  [red]Error: {e}[/red]")

# ── 4. Try every index name that exists ───────────────────────────────────────
console.print(f"\n[cyan]Trying every index name:[/cyan]")
for idx in indexes:
    name = idx.get("name")
    status = idx.get("status")
    try:
        r = list(col.aggregate([
            {"$vectorSearch": {
                "index": name,
                "path": "embedding",
                "queryVector": zero,
                "numCandidates": 10,
                "limit": 3,
            }},
            {"$project": {"name": 1}},
        ]))
        console.print(f"  index={name!r} status={status} → [green]{len(r)} results[/green]")
    except Exception as e:
        console.print(f"  index={name!r} status={status} → [red]error: {e}[/red]")

console.rule("[bold]Done[/bold]")
