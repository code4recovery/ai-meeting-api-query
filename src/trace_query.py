#!/usr/bin/env python3
"""
trace_query.py — Trace every step of a RAG query to find where results are lost.
Standalone — no src imports needed.

Usage:
    python trace_query.py
    python trace_query.py --query "monday morning meeting"
"""
import sys, os, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Load .env manually
env = {}
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

def cfg(k, d=""): return env.get(k) or os.environ.get(k, d)

GEMINI_API_KEY     = cfg("GEMINI_API_KEY")
GEMINI_EMBED_MODEL = cfg("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_LLM_MODEL   = cfg("GEMINI_MODEL", "gemini-1.5-flash")
MONGODB_URI        = cfg("MONGODB_URI")
MONGODB_DB         = cfg("MONGODB_DB", "aa_meetings")
MONGODB_COL        = cfg("MONGODB_COLLECTION", "meetings")
VECTOR_INDEX       = cfg("VECTOR_INDEX_NAME", "aa_meetings_vector_index")

from rich.console import Console
from rich.panel import Panel
console = Console()

parser = argparse.ArgumentParser()
parser.add_argument("--query", default="monday morning meeting Long Beach")
args = parser.parse_args()

console.rule("[bold blue]RAG Query Tracer[/bold blue]")
console.print(f"Query: [bold]{args.query}[/bold]")
console.print(f"Index: {VECTOR_INDEX}  |  DB: {MONGODB_DB}.{MONGODB_COL}")

# ── Step 1: Connect ───────────────────────────────────────────────────────────
console.print("\n[cyan]Step 1: MongoDB connection[/cyan]")
from pymongo import MongoClient
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
col = client[MONGODB_DB][MONGODB_COL]
total = col.count_documents({})
console.print(f"  Total docs: {total}")

# ── Step 2: Check a sample doc ────────────────────────────────────────────────
console.print("\n[cyan]Step 2: Sample document[/cyan]")
doc = col.find_one({}, {"embedding": 1, "name": 1, "day": 1, "region": 1,
                        "types": 1, "attendance_option": 1})
if not doc:
    console.print("[red]No documents found![/red]"); sys.exit(1)

emb = doc.get("embedding", [])
console.print(f"  name:              {doc.get('name')}")
console.print(f"  day:               {doc.get('day')}  (type: {type(doc.get('day')).__name__})")
console.print(f"  region:            {doc.get('region')}  (type: {type(doc.get('region')).__name__})")
console.print(f"  types:             {doc.get('types')}  (type: {type(doc.get('types')).__name__})")
console.print(f"  attendance_option: {doc.get('attendance_option')}")
console.print(f"  embedding dims:    {len(emb)}")

stored_dims = len(emb)

# ── Step 3: Embed the query ───────────────────────────────────────────────────
console.print("\n[cyan]Step 3: Embed query[/cyan]")
from google import genai
from google.genai import types as gtypes
gclient = genai.Client(api_key=GEMINI_API_KEY)
result = gclient.models.embed_content(
    model=GEMINI_EMBED_MODEL,
    contents=args.query,
    config=gtypes.EmbedContentConfig(task_type="retrieval_query"),
)
qvec = result.embeddings[0].values
console.print(f"  Query dims: {len(qvec)}")

if len(qvec) != stored_dims:
    console.print(f"[red]  DIM MISMATCH: query={len(qvec)} stored={stored_dims} — re-ingest with correct model[/red]")
    sys.exit(1)

# ── Step 4: Raw $vectorSearch — NO filter ────────────────────────────────────
console.print("\n[cyan]Step 4: $vectorSearch — no filter[/cyan]")
try:
    pipeline = [
        {"$vectorSearch": {
            "index": VECTOR_INDEX,
            "path": "embedding",
            "queryVector": qvec,
            "numCandidates": 200,
            "limit": 5,
        }},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0, "embedding_text": 0, "_raw": 0}},
    ]
    results = list(col.aggregate(pipeline))
    console.print(f"  Results: [bold]{len(results)}[/bold]")
    for r in results:
        console.print(f"    name={r.get('name')!r}  day={r.get('day')}  "
                      f"region={r.get('region')!r}  score={r.get('score','?'):.4f}")
    if not results:
        console.print("  [red]0 results — index not READY or name wrong[/red]")
        console.print(f"  Expected index name: {VECTOR_INDEX!r}")
        console.print("  Check Atlas UI: Search Indexes → status must be READY")
        sys.exit(1)
except Exception as e:
    console.print(f"  [red]$vectorSearch failed: {e}[/red]")
    sys.exit(1)

# ── Step 5: Check what fields are actually stored ─────────────────────────────
console.print("\n[cyan]Step 5: All fields in a returned document[/cyan]")
r0 = {k: v for k, v in results[0].items() if k != "embedding"}
for k, v in r0.items():
    console.print(f"  {k}: {repr(v)[:80]}")

# ── Step 6: Test the RAG prompt ───────────────────────────────────────────────
console.print("\n[cyan]Step 6: Gemini LLM answer[/cyan]")
def build_context(meetings):
    parts = []
    for i, m in enumerate(meetings, 1):
        lines = [f"[Meeting {i}]"]
        for field in ["name","day_text","day","time_formatted","time",
                      "location","formatted_address","region","types","notes"]:
            v = m.get(field)
            if v not in (None, "", []):
                lines.append(f"  {field}: {v}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)

context = build_context(results)
prompt = f"""You are an AA meetings assistant. Answer using ONLY the meetings below.

--- MEETINGS ---
{context}
--- END ---

Question: {args.query}"""

resp = gclient.models.generate_content(
    model=GEMINI_LLM_MODEL,
    contents=prompt,
    config=gtypes.GenerateContentConfig(max_output_tokens=512, temperature=0.2),
)
console.print(Panel(resp.text.strip(), title="LLM Answer", border_style="green"))

console.rule("[bold green]Trace complete[/bold green]")
