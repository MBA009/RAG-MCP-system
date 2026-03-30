"""
mcp_server/server.py
─────────────────────
Exposes the Second Brain as an MCP server.
Claude (or any MCP client) can call these tools:

  search_brain(query, top_k?)       → ranked chunks with citations
  add_document(path_or_url)         → ingest a new source on the fly
  list_sources()                    → all ingested documents
  delete_source(file_hash)          → remove a document

Run this server with:
    python -m mcp_server.server

Then add it to Claude Desktop's config:
    {
      "mcpServers": {
        "second-brain": {
          "command": "python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "/path/to/second-brain"
        }
      }
    }
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from retrieval.retriever  import Retriever
from ingestion.ingestor   import Ingestor
from storage.store        import TextStore
from config               import MCP_SERVER_NAME

# ── Initialise shared instances ───────────────────────────────────────────────
server     = Server(MCP_SERVER_NAME)
retriever  = Retriever()
ingestor   = Ingestor()
text_store = TextStore()


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_brain",
            description=(
                "Search your personal Second Brain knowledge base. "
                "Returns the most relevant passages from your notes, PDFs, "
                "bookmarks, code files and web pages."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="add_document",
            description=(
                "Add a new document or URL to your Second Brain. "
                "Accepts a local file path (PDF, Markdown, DOCX, code file) "
                "or a web URL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path_or_url": {
                        "type": "string",
                        "description": "Absolute file path or https:// URL",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Re-ingest even if already present",
                        "default": False,
                    },
                },
                "required": ["path_or_url"],
            },
        ),
        types.Tool(
            name="list_sources",
            description="List all documents currently in your Second Brain.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="delete_source",
            description="Remove a document from your Second Brain by its file hash.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_hash": {
                        "type": "string",
                        "description": "The file_hash returned by list_sources",
                    },
                },
                "required": ["file_hash"],
            },
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tool handlers
# ─────────────────────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── search_brain ──────────────────────────────────────────────────────────
    if name == "search_brain":
        query  = arguments["query"]
        top_k  = int(arguments.get("top_k", 5))
        chunks = retriever.retrieve(query, top_k=top_k)

        if not chunks:
            return [types.TextContent(
                type="text",
                text="No relevant passages found in your Second Brain.",
            )]

        lines = [f"Found {len(chunks)} relevant passages:\n"]
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[{i}] **{c.get('title', 'Untitled')}** "
                f"(score: {c.get('score', 0):.3f})\n"
                f"Source: {c.get('source_path', 'unknown')}\n"
                f"Type: {c.get('source_type', '?')}\n\n"
                f"{c.get('text', '')}\n"
                f"{'─' * 60}"
            )
        return [types.TextContent(type="text", text="\n".join(lines))]

    # ── add_document ──────────────────────────────────────────────────────────
    elif name == "add_document":
        target = arguments["path_or_url"]
        force  = bool(arguments.get("force", False))

        if target.startswith("http://") or target.startswith("https://"):
            result = ingestor.ingest_url(target, force=force)
        else:
            result = ingestor.ingest_file(Path(target), force=force)

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    # ── list_sources ──────────────────────────────────────────────────────────
    elif name == "list_sources":
        sources = text_store.list_sources()
        if not sources:
            return [types.TextContent(type="text", text="No sources ingested yet.")]
        lines = [f"Your Second Brain contains {len(sources)} source(s):\n"]
        for s in sources:
            lines.append(
                f"• {s['title']} [{s['source_type']}]\n"
                f"  Path: {s['source_path']}\n"
                f"  Ingested: {s['ingested_at']}\n"
                f"  Hash: {s['file_hash'][:12]}…"
            )
        return [types.TextContent(type="text", text="\n".join(lines))]

    # ── delete_source ─────────────────────────────────────────────────────────
    elif name == "delete_source":
        file_hash = arguments["file_hash"]
        ingestor.delete_source(file_hash)
        return [types.TextContent(
            type="text",
            text=f"Deleted all chunks for source hash {file_hash[:12]}…",
        )]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
