"""
cli.py
───────
Management CLI for your Second Brain.

Commands:
    python cli.py ingest <path>          ingest a file or directory
    python cli.py ingest-url <url>       ingest a web page
    python cli.py list                   list all sources
    python cli.py delete <file_hash>     remove a source
    python cli.py stats                  show stats
    python cli.py watch                  watch inbox dir for new files
    python cli.py chat                   start the chat interface
    python cli.py serve                  start the MCP server
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import typer
from rich.console import Console
from rich         import print as rprint

app     = typer.Typer(help="🧠 Second Brain — personal RAG knowledge base")
console = Console()


@app.command()
def ingest(
    path:  str  = typer.Argument(..., help="File or directory path to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest existing docs"),
):
    """Ingest a file or directory into your Second Brain."""
    from ingestion.ingestor import Ingestor
    ingestor = Ingestor()
    p = Path(path)
    if p.is_dir():
        console.print(f"[bold]Ingesting directory:[/bold] {p}")
        results = ingestor.ingest_directory(p, force=force)
        total   = sum(r.get("chunks", 0) for r in results)
        console.print(f"\n[green]Done. {len(results)} files, {total} total chunks.[/green]")
    elif p.is_file():
        console.print(f"[bold]Ingesting:[/bold] {p.name}")
        result = ingestor.ingest_file(p, force=force)
        console.print(f"[green]{result}[/green]")
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)


@app.command("ingest-url")
def ingest_url(
    url:   str  = typer.Argument(..., help="URL to ingest"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Ingest a web page into your Second Brain."""
    from ingestion.ingestor import Ingestor
    result = Ingestor().ingest_url(url, force=force)
    console.print(f"[green]{result}[/green]")


@app.command("list")
def list_sources():
    """List all sources in your Second Brain."""
    from storage.store import TextStore
    from rich.table    import Table
    sources = TextStore().list_sources()
    if not sources:
        console.print("[yellow]No sources ingested yet.[/yellow]")
        return
    table = Table(title=f"📁 {len(sources)} source(s)")
    table.add_column("Title");      table.add_column("Type")
    table.add_column("Ingested");   table.add_column("Hash (12)", style="dim")
    for s in sources:
        table.add_row(s["title"], s["source_type"],
                      s["ingested_at"][:10], s["file_hash"][:12])
    console.print(table)


@app.command()
def delete(file_hash: str = typer.Argument(..., help="File hash from 'list' command")):
    """Remove a source document and all its chunks."""
    from ingestion.ingestor import Ingestor
    Ingestor().delete_source(file_hash)
    console.print(f"[green]Deleted {file_hash[:12]}…[/green]")


@app.command()
def stats():
    """Show knowledge base statistics."""
    from ingestion.ingestor import Ingestor
    s = Ingestor().stats()
    console.print(f"[bold]Second Brain stats[/bold]")
    console.print(f"  Sources:       {s['sources']}")
    console.print(f"  Total chunks:  {s['total_chunks']}")
    console.print(f"  Vector points: {s['vector_points']}")


@app.command()
def watch():
    """Watch the inbox directory and auto-ingest new files."""
    from ingestion.ingestor import watch_directory
    from config             import WATCH_DIR
    console.print(f"[bold]Watching:[/bold] {WATCH_DIR}")
    watch_directory(WATCH_DIR)


@app.command()
def chat():
    """Start the interactive chat interface."""
    from interface.chat import chat as start_chat
    start_chat()


@app.command()
def serve():
    """Start the MCP server (connect to Claude Desktop or any MCP client)."""
    import asyncio
    from mcp_server.server import main
    console.print("[bold]Starting Second Brain MCP server…[/bold]")
    asyncio.run(main())

@app.command()
def web():
    """Start the web UI (open http://localhost:8000 in your browser)."""
    import subprocess
    console.print("[bold]Starting Second Brain web UI at http://localhost:8000[/bold]")
    subprocess.run(["uvicorn", "api:app", "--port", "8000"])


@app.command("ingest-youtube")
def ingest_youtube_cmd(url: str = typer.Argument(..., help="YouTube URL or video ID")):
    """Ingest a YouTube video transcript."""
    from ingestion.sources  import parse_youtube
    from ingestion.ingestor import Ingestor
    console.print(f"[bold]Fetching transcript:[/bold] {url}")
    text, meta = parse_youtube(url)
    result = Ingestor()._ingest(text, meta, force=False)
    console.print(f"[green]{result}[/green]")


@app.command("ingest-notion")
def ingest_notion_cmd():
    """Sync all Notion pages (requires NOTION_TOKEN env var)."""
    from ingestion.sources  import iter_notion_pages
    from ingestion.ingestor import Ingestor
    ing = Ingestor()
    total, ingested = 0, 0
    for text, meta in iter_notion_pages():
        r = ing._ingest(text, meta, force=False)
        total += 1
        if r["status"] == "ingested":
            ingested += 1
            console.print(f"  [green]✓[/green] {meta['title']} ({r['chunks']} chunks)")
        else:
            console.print(f"  [dim]skipped: {meta['title']}[/dim]")
    console.print(f"\n[bold]Done.[/bold] {ingested}/{total} pages ingested.")

if __name__ == "__main__":
    app()


