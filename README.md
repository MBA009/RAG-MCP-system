# 🧠 Second Brain

A personal AI-powered knowledge base. Ingest your PDFs, notes, bookmarks, code files, and YouTube videos — then chat with them using Claude.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green) ![Claude](https://img.shields.io/badge/Claude-Sonnet-purple)

---

## What it does

- **Ingest** any document: PDFs, Markdown, DOCX, code files, web URLs, YouTube transcripts, Notion pages
- **Search** using hybrid retrieval — semantic vector search + keyword search, merged with a real ML re-ranker
- **Chat** with your documents through a web UI, answers come with source citations
- **Expose** your knowledge base as an MCP tool so Claude Desktop can search it mid-conversation

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/second-brain.git
cd second-brain
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
EMBEDDING_BACKEND=local
```

`EMBEDDING_BACKEND=local` uses a free offline model (recommended to start).
If you have an OpenAI key and want higher quality embeddings:
```
EMBEDDING_BACKEND=openai
OPENAI_API_KEY=sk-...
```

Optional:
```
NOTION_TOKEN=secret_...
```

### 3. Ingest your documents

```powershell
# Single file
python cli.py ingest ~/Documents/notes.pdf

# Entire folder (recursive)
python cli.py ingest ~/Documents/notes/

# Web page
python cli.py ingest-url https://en.wikipedia.org/wiki/Retrieval-augmented_generation

# YouTube video (fetches transcript automatically, no API key needed)
python cli.py ingest-youtube https://www.youtube.com/watch?v=VIDEO_ID

# All Notion pages (requires NOTION_TOKEN in .env)
python cli.py ingest-notion
```

### 4. Start the web UI

```powershell
python cli.py web
```

Open **http://localhost:8000** in your browser.

---

## Web UI features

- Chat with your documents — answers include clickable source citations
- Add sources from the sidebar: URL, YouTube, file upload, or Notion sync
- Hover over any source and click ✕ to remove it
- Header shows total docs and chunks indexed

---

## All CLI commands

| Command | What it does |
|---|---|
| `python cli.py ingest <path>` | Ingest a file or directory |
| `python cli.py ingest-url <url>` | Ingest a web page |
| `python cli.py ingest-youtube <url>` | Ingest a YouTube transcript |
| `python cli.py ingest-notion` | Sync all Notion pages |
| `python cli.py list` | List all ingested sources |
| `python cli.py delete <hash>` | Delete a source by its hash |
| `python cli.py stats` | Show total sources and chunks |
| `python cli.py watch` | Watch inbox folder and auto-ingest new files |
| `python cli.py chat` | Terminal chat interface |
| `python cli.py web` | Start the web UI |
| `python cli.py serve` | Start the MCP server for Claude Desktop |

---

## Claude Desktop integration (MCP)

Add to your Claude Desktop config:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**Mac:** `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "python",
      "args": ["cli.py", "serve"],
      "cwd": "C:\\path\\to\\second-brain"
    }
  }
}
```

Restart Claude Desktop. Claude can now call `search_brain`, `add_document`, `list_sources`, and `delete_source` as tools mid-conversation.

---

## Key configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_BACKEND` | `"local"` | `"local"` or `"openai"` |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `TOP_K_FINAL` | `5` | Results returned per query |
| `CLAUDE_MODEL` | `claude-sonnet-4-...` | Claude model used for chat |

---

## Supported file types

PDF, DOCX, Markdown, plain text, Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, HTML, web URLs, YouTube, Notion, Obsidian vaults

---

## Tech stack

| Tool | Role |
|---|---|
| [Qdrant](https://qdrant.tech/) | Local vector database |
| [SQLite FTS5](https://www.sqlite.org/fts5.html) | Full-text keyword search |
| [sentence-transformers](https://www.sbert.net/) | Embeddings + cross-encoder re-ranking |
| [FastAPI](https://fastapi.tiangolo.com/) | Web API |
| [Anthropic Claude](https://www.anthropic.com/) | Language model |
| [MCP SDK](https://github.com/anthropics/mcp) | Claude Desktop tool integration |