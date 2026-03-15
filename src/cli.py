#!/usr/bin/env python3
"""
cli.py — Interactive CLI for querying AA meetings via RAG.

Usage:
    python src/cli.py
"""
import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.rule import Rule
from rich import box

from rag import run_rag

console = Console()

BANNER = """
[bold blue]╔══════════════════════════════════════════════╗
║   AA Meetings Finder — Powered by Gemini    ║
║   Type  [green]help[/green]  for tips,  [red]exit[/red]  to quit       ║
╚══════════════════════════════════════════════╝[/bold blue]
"""

HELP_TEXT = """
[bold]Example queries:[/bold]
  • Are there Big Book meetings on Tuesday evenings?
  • Find women-only meetings in Long Beach on weekends
  • What online meetings are available Friday mornings?
  • Show open discussion meetings on Monday

[bold]Commands:[/bold]
  [green]help[/green]   — show this message
  [green]clear[/green]  — clear screen
  [red]exit[/red]   — quit
"""

TYPE_LABELS = {
    "12x12":"12 Steps & 12 Traditions","1HR":"1 Hour","A":"Atheist/Agnostic",
    "ABSI":"As Bill Sees It","ASL":"Sign Language","B":"Beginners","BE":"Newcomer",
    "BS":"Book Study","C":"Closed","CAN":"Candlelight","CF":"Child Friendly",
    "D":"Discussion","DR":"Daily Reflections","EN":"Español","InP":"In Person",
    "LGBTQ":"LGBTQ+","LIT":"Literature","LS":"Living Sober","M":"Men",
    "MED":"Meditation","NB":"Non-Binary","O":"Open","ONL":"Online",
    "Part":"Participation","QA":"Q&A","S":"Spanish","SEN":"Seniors",
    "SP":"Speaker","SS":"Step Study","ST":"Step","T":"Traditions",
    "TC":"Temp Closed","VM":"Virtual","W":"Women","X":"Open to All",
    "XT":"Open to All+","Y":"Young People",
}

def _type_label(code: str) -> str:
    return TYPE_LABELS.get(code, code)


def display_meetings_table(meetings: list, title: str = "Matching Meetings"):
    if not meetings:
        console.print("[yellow]No meetings matched.[/yellow]")
        return

    table = Table(
        title=f"[bold]{title}[/bold]  ({len(meetings)} meeting{'s' if len(meetings) != 1 else ''})",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#",              style="dim",     width=3,  justify="right")
    table.add_column("Name",           style="bold white", max_width=28)
    table.add_column("Day / Time",     style="yellow",  max_width=16)
    table.add_column("Location",       style="green",   max_width=26)
    table.add_column("Types",          style="magenta", max_width=22)
    table.add_column("Attendance",     style="cyan",    max_width=12)
    table.add_column("Score",          style="dim",     width=6, justify="right")

    for i, m in enumerate(meetings, 1):
        day   = m.get("day_text", "") or ""
        time_ = m.get("time_formatted") or m.get("time", "")
        day_time = f"{day}\n{time_}" if day and time_ else day or time_

        loc  = (m.get("location") or "").strip()
        addr = (m.get("formatted_address") or "").strip()
        if loc and addr and loc != addr:
            location_str = f"{loc}\n{addr[:22]}"
        else:
            location_str = loc or addr

        types = m.get("types", [])
        if isinstance(types, str):
            types = [types]
        types_str = "\n".join(_type_label(t) for t in types[:5])

        att   = (m.get("attendance_option") or "").replace("_", " ")
        score = m.get("score")
        score_str = f"{score:.3f}" if score is not None else ""

        table.add_row(
            str(i),
            (m.get("name") or "")[:28],
            day_time,
            location_str[:26],
            types_str,
            att,
            score_str,
        )

    console.print()
    console.print(table)


def startup_check():
    from config import settings
    issues = []
    if not settings.gemini_api_key or "your_" in settings.gemini_api_key:
        issues.append("GEMINI_API_KEY not set in .env")
    if not settings.mongodb_uri or "your" in settings.mongodb_uri:
        issues.append("MONGODB_URI not set in .env")
    if issues:
        for i in issues:
            console.print(f"[red]✗ {i}[/red]")
        sys.exit(1)
    console.print(f"[dim]  Model:      {settings.gemini_model}[/dim]")
    console.print(f"[dim]  Embeddings: {settings.gemini_embedding_model}[/dim]")
    console.print(f"[dim]  MongoDB:    {settings.mongodb_db}.{settings.mongodb_collection}[/dim]")


def chat_loop():
    console.print(BANNER)
    startup_check()
    console.print("[dim]Ready. Ask anything about AA meetings.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! Keep coming back.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye! Keep coming back.[/dim]")
            break
        if user_input.lower() == "help":
            console.print(HELP_TEXT)
            continue
        if user_input.lower() == "clear":
            console.clear()
            console.print(BANNER)
            continue

        # ── Run RAG ──────────────────────────────────────────────────────
        with console.status("[bold green]Searching meetings and generating answer...[/bold green]"):
            try:
                answer, cited_meetings = run_rag(user_input)
            except Exception:
                console.print_exception(show_locals=False)
                continue

        # ── 1. Answer first ───────────────────────────────────────────────
        console.print(
            Panel(
                Markdown(answer),
                title="[bold green]Assistant[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # ── 2. Table of cited meetings below the answer ───────────────────
        if cited_meetings:
            display_meetings_table(
                cited_meetings,
                title="Meetings referenced above",
            )
        else:
            console.print("[yellow]No meetings were retrieved from the database.[/yellow]")

        console.print()


if __name__ == "__main__":
    chat_loop()
