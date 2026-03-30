"""
interface/chat.py
──────────────────
Interactive CLI chat loop.
Each user message triggers retrieval → context injection → Claude generation.
Citations are printed alongside each response.

Usage:
    python -m interface.chat
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from rich.console    import Console
from rich.markdown   import Markdown
from rich.panel      import Panel
from rich.prompt     import Prompt
from rich.table      import Table
from rich            import print as rprint

from retrieval.retriever import Retriever
from config              import ANTHROPIC_API_KEY, CLAUDE_MODEL, TOP_K_FINAL

console   = Console()
retriever = Retriever()
client    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a personal knowledge assistant with access to the user's Second Brain — 
a curated collection of their notes, PDFs, bookmarks, and documents.

When answering questions:
1. Base your answer primarily on the retrieved context passages provided.
2. Always cite your sources using [1], [2], etc. referencing the passages below.
3. If the context doesn't contain enough information, say so clearly and share what you do know.
4. Be concise but thorough.
5. If asked about something clearly outside the knowledge base, answer from general knowledge and note it.

The retrieved context will be prepended to each user message."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    if not chunks:
        return "[No relevant context found in your Second Brain]\n"
    lines = ["### Retrieved context from your Second Brain:\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] Source: {c.get('title', 'Untitled')} "
            f"({c.get('source_type', '?')}) — score {c.get('score', 0):.3f}\n"
            f"{c['text']}\n"
        )
    return "\n".join(lines)


def show_sources(chunks: list[dict]):
    """Print a compact sources table."""
    if not chunks:
        return
    table = Table(title="📚 Sources used", show_header=True, header_style="bold cyan")
    table.add_column("#",      width=3)
    table.add_column("Title",  style="bold")
    table.add_column("Type",   width=8)
    table.add_column("Score",  width=7)
    table.add_column("Path",   style="dim", overflow="fold")
    for i, c in enumerate(chunks, 1):
        table.add_row(
            str(i),
            c.get("title", "Untitled")[:40],
            c.get("source_type", "?"),
            f"{c.get('score', 0):.3f}",
            str(c.get("source_path", ""))[:60],
        )
    console.print(table)


def chat():
    history: list[dict] = []
    console.print(Panel(
        "[bold]🧠 Second Brain[/bold]\n"
        "Ask anything. Your knowledge base is searched on every message.\n"
        "Type [bold cyan]sources[/bold cyan] to list ingested documents, "
        "[bold cyan]exit[/bold cyan] to quit.",
        expand=False,
    ))

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower() == "sources":
            from storage.store import TextStore
            ts      = TextStore()
            sources = ts.list_sources()
            table   = Table(title=f"📁 {len(sources)} source(s) in your Second Brain")
            table.add_column("Title");  table.add_column("Type"); table.add_column("Ingested")
            for s in sources:
                table.add_row(s["title"], s["source_type"], s["ingested_at"][:10])
            console.print(table)
            continue

        # ── Retrieve ──────────────────────────────────────────────────────────
        with console.status("[dim]🔍 Searching knowledge base…[/dim]"):
            chunks = retriever.retrieve(user_input, top_k=TOP_K_FINAL)

        context     = format_context(chunks)
        augmented   = f"{context}\n### User question:\n{user_input}"

        history.append({"role": "user", "content": augmented})

        # ── Generate (stream) ─────────────────────────────────────────────────
        console.print("\n[bold blue]Assistant[/bold blue]")
        full_response = ""
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=history,
        ) as stream:
            for text in stream.text_stream:
                console.print(text, end="", highlight=False)
                full_response += text

        console.print()   # newline after streaming

        # Add *clean* assistant message to history (no context prefix)
        history.append({"role": "assistant", "content": full_response})

        # Replace the user turn in history with the clean version (no ctx prefix)
        history[-2]["content"] = user_input

        # ── Show sources ──────────────────────────────────────────────────────
        if chunks:
            console.print()
            show_sources(chunks)


if __name__ == "__main__":
    chat()
