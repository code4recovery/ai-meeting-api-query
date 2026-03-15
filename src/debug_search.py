#!/usr/bin/env python3
"""
debug_search.py — Standalone diagnostic. No 'src' imports needed.

Usage (from anywhere):
    python debug_search.py
    python debug_search.py --env /path/to/.env
"""
# Add this to the top of ingest_local.py, before MongoClient is called
import pymongo
import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']
import sys
import os
import argparse
from pathlib import Path

# ── locate .env ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--env", default=None, help="Path to .env file")
args = parser.parse_args()

if args.env:
    env_path = Path(args.env)
else:
    # Search: current dir, then script's dir
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
        if candidate.exists():
            env_path = candidate
            break
    else:
        env_path = Path(".env")   # fallback — will fail gracefully below

# ── load .env manually (no dotenv dependency needed) ─────────────────────────
env_vars = {}
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip('"').strip("'")
    print(f"Loaded .env from: {env_path}")
else:
    print(f"WARNING: .env not found at {env_path} — will use environment variables")

def cfg(key, default=""):
    return env_vars.get(key) or os.environ.get(key, default)

GEMINI_API_KEY      = cfg("GEMINI_API_KEY")
GEMINI_EMBED_MODEL  = cfg("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_LLM_MODEL    = cfg("GEMINI_MODEL", "gemini-1.5-flash")
MONGODB_URI         = cfg("MONGODB_URI")
MONGODB_DB          = cfg("MONGODB_DB", "aa_meetings")
MONGODB_COLLECTION  = cfg("MONGODB_COLLECTION", "meetings")
VECTOR_INDEX_NAME   = cfg("VECTOR_INDEX_NAME", "aa_meetings_vector_index")
EMBEDDING_DIMS      = int(cfg("EMBEDDING_DIMENSIONS", "3072"))

# ── check deps ────────────────────────────────────────────────────────────────
missing = []
for pkg in ["pymongo", "rich", "google.genai"]:
    try:
        __import__(pkg.replace(".", "."))
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"Missing packages: {missing}")
    print("Run: pip install pymongo rich google-genai")
    sys.exit(1)

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ── main ──────────────────────────────────────────────────────────────────────

def _resolve_srv(uri: str) -> str:
    """Resolve mongodb+srv:// via Google DNS to bypass broken system resolvers."""
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


def main():
    console.rule("[bold blue]AA Meetings — Vector Search Debugger[/bold blue]")

    # ── Config summary ────────────────────────────────────────────────────
    console.print("\n[bold]Config:[/bold]")
    console.print(f"  GEMINI_API_KEY:      {'set ✓' if GEMINI_API_KEY else '[red]NOT SET[/red]'}")
    console.print(f"  GEMINI_EMBED_MODEL:  {GEMINI_EMBED_MODEL}")
    console.print(f"  GEMINI_LLM_MODEL:    {GEMINI_LLM_MODEL}")
    console.print(f"  MONGODB_URI:         {'set ✓' if MONGODB_URI else '[red]NOT SET[/red]'}")
    console.print(f"  MONGODB_DB:          {MONGODB_DB}")
    console.print(f"  MONGODB_COLLECTION:  {MONGODB_COLLECTION}")
    console.print(f"  VECTOR_INDEX_NAME:   {VECTOR_INDEX_NAME}")
    console.print(f"  EMBEDDING_DIMS:      {EMBEDDING_DIMS}")

    if not GEMINI_API_KEY or not MONGODB_URI:
        console.print("\n[red]Fix missing config above, then re-run.[/red]")
        sys.exit(1)

    # ── Step 1: MongoDB connection ────────────────────────────────────────
    console.print("\n[cyan]Step 1: MongoDB connection...[/cyan]")
    from pymongo import MongoClient
    try:
        resolved = _resolve_srv(MONGODB_URI)
        if resolved != MONGODB_URI:
            console.print(f"  SRV resolved to standard URI")
        client = MongoClient(resolved, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        console.print("  [green]✓ Connected[/green]")
    except Exception as e:
        console.print(f"  [red]✗ Connection failed: {e}[/red]")
        sys.exit(1)

    col = client[MONGODB_DB][MONGODB_COLLECTION]

    # ── Step 2: Collection contents ───────────────────────────────────────
    console.print("\n[cyan]Step 2: Collection contents...[/cyan]")
    total = col.count_documents({})
    console.print(f"  Total documents: [bold]{total}[/bold]")
    if total == 0:
        console.print("  [red]✗ Empty — run ingest_local.py first[/red]")
        sys.exit(1)

    doc = col.find_one({"embedding": {"$exists": True}})
    if doc is None:
        console.print("  [red]✗ No documents have an 'embedding' field — re-run ingest[/red]")
        sys.exit(1)

    stored_dims = len(doc["embedding"])
    no_emb = col.count_documents({"embedding": {"$exists": False}})
    console.print(f"  Stored embedding dims:   [bold]{stored_dims}[/bold]")
    console.print(f"  Docs without embedding:  [bold]{no_emb}[/bold]")

    if stored_dims != EMBEDDING_DIMS:
        console.print(
            f"  [red]✗ Dim mismatch: stored={stored_dims} vs EMBEDDING_DIMENSIONS={EMBEDDING_DIMS}\n"
            f"    Fix: set EMBEDDING_DIMENSIONS={stored_dims} in .env\n"
            f"    AND recreate the Atlas index with numDimensions={stored_dims}[/red]"
        )
    else:
        console.print(f"  [green]✓ Dims match ({stored_dims})[/green]")

    # ── Step 3: Atlas search indexes ──────────────────────────────────────
    console.print("\n[cyan]Step 3: Atlas search indexes...[/cyan]")
    index_ready = False
    try:
        indexes = list(col.list_search_indexes())
        if not indexes:
            console.print("  [red]✗ No search indexes found — create one in Atlas UI[/red]")
        for idx in indexes:
            name   = idx.get("name", "?")
            status = idx.get("status", "?")
            color  = "green" if status == "READY" else "yellow"
            match  = "✓" if name == VECTOR_INDEX_NAME else "✗ NAME MISMATCH"
            console.print(f"  [{color}]{match} name={name!r}  status={status}[/{color}]")

            if name == VECTOR_INDEX_NAME and status == "READY":
                index_ready = True

            defn   = idx.get("latestDefinition") or idx.get("definition") or {}
            for f in defn.get("fields", []):
                if f.get("type") == "vector":
                    ndims = f.get("numDimensions")
                    sim   = f.get("similarity", "?")
                    dim_ok = int(ndims or 0) == stored_dims
                    dc = "green" if dim_ok else "red"
                    console.print(
                        f"    vector path={f.get('path')!r}  "
                        f"[{dc}]numDimensions={ndims}[/{dc}]  similarity={sim}"
                    )
                    if not dim_ok:
                        console.print(
                            f"    [red]✗ Index dim={ndims} != stored dim={stored_dims}\n"
                            f"      Delete and recreate the index with numDimensions={stored_dims}[/red]"
                        )
                else:
                    console.print(f"    filter path={f.get('path')!r}")

        if not index_ready:
            console.print(
                f"\n  [yellow]Index not READY — either still building or wrong name.\n"
                f"  Expected name: {VECTOR_INDEX_NAME!r}\n"
                f"  If status is BUILDING, wait 1-3 min and re-run.[/yellow]"
            )
    except Exception as e:
        console.print(f"  [yellow]Could not list indexes (Atlas tier may not support this): {e}[/yellow]")

    # ── Step 4: Raw $vectorSearch with zero vector ────────────────────────
    console.print("\n[cyan]Step 4: $vectorSearch smoke test (zero vector)...[/cyan]")
    zero_vec = [0.0] * stored_dims
    try:
        results = list(col.aggregate([
            {"$vectorSearch": {
                "index":         VECTOR_INDEX_NAME,
                "path":          "embedding",
                "queryVector":   zero_vec,
                "numCandidates": 150,
                "limit":         3,
            }},
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0, "embedding_text": 0, "_raw": 0}},
        ]))
        console.print(f"  Results: [bold]{len(results)}[/bold]")
        for r in results:
            console.print(f"    {r.get('name','?')}  score={r.get('score','?')}")
        if not results:
            console.print(
                "  [yellow]0 results with zero vector — index may still be building,\n"
                f"  or index name {VECTOR_INDEX_NAME!r} doesn't match Atlas.[/yellow]"
            )
    except Exception as e:
        console.print(f"  [red]✗ $vectorSearch failed: {e}[/red]")

    # ── Step 5: Live embed + search ───────────────────────────────────────
    console.print("\n[cyan]Step 5: Live Gemini embed + vector search...[/cyan]")
    try:
        from google import genai
        from google.genai import types as genai_types

        gclient = genai.Client(api_key=GEMINI_API_KEY)
        test_query = "Monday morning meeting Long Beach"
        result = gclient.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=test_query,
            config=genai_types.EmbedContentConfig(task_type="retrieval_query"),
        )
        qvec = result.embeddings[0].values
        console.print(f"  Query embedded OK  dims={len(qvec)}")

        results = list(col.aggregate([
            {"$vectorSearch": {
                "index":         VECTOR_INDEX_NAME,
                "path":          "embedding",
                "queryVector":   qvec,
                "numCandidates": 150,
                "limit":         5,
            }},
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0, "embedding_text": 0, "_raw": 0}},
        ]))

        console.print(f"  Results: [bold]{len(results)}[/bold]")
        if results:
            t = Table(box=box.SIMPLE, header_style="bold cyan")
            t.add_column("Name");   t.add_column("Day")
            t.add_column("Time");   t.add_column("Region"); t.add_column("Score")
            for r in results:
                t.add_row(
                    (r.get("name") or "")[:30],
                    r.get("day_text", ""),
                    r.get("time_formatted") or r.get("time", ""),
                    r.get("region", ""),
                    f"{r.get('score', 0):.4f}",
                )
            console.print(t)
        else:
            console.print("  [red]Still 0 results — check index status and dims above[/red]")

    except Exception as e:
        console.print(f"  [red]✗ Failed: {e}[/red]")
        import traceback; traceback.print_exc()

    console.rule("[bold]Done[/bold]")


if __name__ == "__main__":
    main()
