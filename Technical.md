# 🧠 Second Brain — Technical Documentation

This document explains every file in the project, how they connect, and the design decisions behind them. Intended for developers who want to understand, extend, or fork the codebase.

---

## Architecture overview

The system has five layers that data flows through linearly:

```
Your files / URLs
      ↓
┌─────────────────────────────────┐
│  INGESTION LAYER                │  parsers.py → chunker.py → embedder.py → ingestor.py
│  Parse → Chunk → Embed → Store  │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  STORAGE LAYER                  │  store.py
│  Qdrant (vectors) + SQLite      │
└─────────────────────────────────┘
      ↓  (on query)
┌─────────────────────────────────┐
│  RETRIEVAL LAYER                │  retriever.py
│  Hybrid search + re-ranking     │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  INTERFACE LAYER                │  api.py + frontend/ OR interface/chat.py
│  Web UI / Terminal / MCP        │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  LLM LAYER                      │  Claude API
│  Context injection + generation │
└─────────────────────────────────┘
```

---

## File-by-file breakdown

### `config.py`

The single source of truth for all settings. Every other file imports from here — nothing is hardcoded anywhere else.

Key settings:
- `EMBEDDING_BACKEND` — switches between OpenAI API and local sentence-transformers
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — controls how documents are split
- `TOP_K_DENSE` / `TOP_K_BM25` / `TOP_K_FINAL` — retrieval pipeline parameters
- `QDRANT_PATH` / `SQLITE_PATH` — where data is stored on disk
- `CLAUDE_MODEL` — which Claude model to use for chat

Uses `python-dotenv` to load a `.env` file so API keys are never hardcoded.

---

### `cli.py`

The command-line entry point. Built with [Typer](https://typer.tiangolo.com/) which turns Python functions decorated with `@app.command()` into CLI commands automatically.

Each command is a thin wrapper that imports and calls the relevant module. The CLI itself contains no business logic.

Commands and what they call:
| CLI command | Calls |
|---|---|
| `ingest` | `ingestion/ingestor.py → Ingestor.ingest_file()` |
| `ingest-url` | `ingestion/ingestor.py → Ingestor.ingest_url()` |
| `ingest-youtube` | `ingestion/sources.py → parse_youtube()` |
| `ingest-notion` | `ingestion/sources.py → iter_notion_pages()` |
| `list` | `storage/store.py → TextStore.list_sources()` |
| `delete` | `storage/store.py → TextStore.list_sources()` + `ingestion/ingestor.py → delete_source()` |
| `stats` | `ingestion/ingestor.py → Ingestor.stats()` |
| `watch` | `ingestion/ingestor.py → watch_directory()` |
| `chat` | `interface/chat.py → chat()` |
| `web` | `subprocess → uvicorn api:app` |
| `serve` | `mcp_server/server.py → main()` |

---

### `api.py`

FastAPI web server. Exposes REST endpoints consumed by the frontend.

All heavy objects (`Retriever`, `Ingestor`, `TextStore`) are initialized **lazily** — only on the first request, not at import time. This is critical on Windows where Qdrant's file lock prevents two processes opening the same database simultaneously.

Endpoints:
| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | Serves `frontend/index.html` |
| `POST` | `/chat/stream` | SSE streaming chat — retrieves context then streams Claude response |
| `GET` | `/sources` | Returns list of all ingested sources |
| `GET` | `/stats` | Returns source count and chunk count |
| `POST` | `/ingest/url` | Ingests a web URL |
| `POST` | `/ingest/youtube` | Ingests a YouTube transcript |
| `POST` | `/ingest/notion` | Syncs all Notion pages |
| `POST` | `/ingest/file` | Accepts file upload, saves to temp dir, ingests |
| `DELETE` | `/source/{hash}` | Deletes a source and all its chunks |

The `/chat/stream` endpoint uses [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) to stream tokens to the browser. It emits three event types:
- `{"type": "sources", "sources": [...]}` — emitted first so the UI shows sources immediately
- `{"type": "token", "text": "..."}` — one per streamed token from Claude
- `{"type": "done"}` — signals end of stream

---

### `frontend/index.html`

Single-file frontend. No build step, no npm, no framework — just HTML, CSS, and vanilla JavaScript in one file.

Key design decisions:
- All chat history is stored in a `chatHistory` variable (deliberately not named `history` to avoid colliding with the browser's built-in `window.history` object)
- Sources are fetched and re-rendered after every ingest operation
- File uploads use `FormData` while all other requests use `application/json`
- The SSE stream is consumed with `fetch()` + `ReadableStream`, not `EventSource`, because `EventSource` doesn't support POST requests
- Suggestions use `data-idx` attributes and `addEventListener` rather than inline `onclick` to avoid issues with special characters in suggestion text

---

### `ingestion/parsers.py`

Knows how to read every supported file type. Contains one function per file type and a `parse_file(path)` dispatcher that routes based on file extension.

Every parser returns `(text: str, metadata: dict)`. The metadata dict always contains: `source_path`, `source_type`, `title`, `file_hash`, `created_at`, `modified_at`.

The `file_hash` is a SHA-256 of the file contents (or URL string for web sources) — used for deduplication. If you try to ingest a file that's already in the database, the ingestor skips it unless `force=True`.

Web parsing (`parse_url`) uses a priority chain to find the main content: Wikipedia article body → `<main>` tag → `<article>` tag → `<div id="content">` → full body. This avoids ingesting nav bars and footers as content.

---

### `ingestion/chunker.py`

Splits long documents into overlapping chunks for storage and retrieval.

Why chunking is necessary: Claude's context window is limited, and vector similarity works better on short focused passages than on whole documents. Chunks of ~500 words strike a balance between enough context and focused retrieval.

The chunker:
1. Splits text into sentences using punctuation markers
2. Greedily packs sentences into chunks up to `CHUNK_CHARS` characters
3. When a chunk is full, carries the last `OVERLAP_CHARS` characters into the next chunk as overlap — so ideas that span a chunk boundary aren't lost

Each chunk inherits all metadata from its parent document, plus `chunk_index` (position in document) and `char_start` (character offset in original text).

---

### `ingestion/embedder.py`

Converts text into dense vectors (lists of floating point numbers that represent meaning).

Two backends behind a common interface (`embed(text)` and `embed_batch(texts)`):

**OpenAI** (`EMBEDDING_BACKEND=openai`): calls `text-embedding-3-small`, produces 1536-dimensional vectors. Better quality, costs ~$0.00002 per 1000 tokens.

**Local** (`EMBEDDING_BACKEND=local`): runs `all-MiniLM-L6-v2` from sentence-transformers entirely on CPU. Produces 384-dimensional vectors. Free, private, slightly lower quality.

The embedder is a singleton — instantiated once and reused across the whole application to avoid reloading model weights on every request.

---

### `ingestion/ingestor.py`

Orchestrates the full ingestion pipeline for a single document:

```
parse_file(path)
    → split_into_chunks(text, meta)
    → embedder.embed_batch(texts)          # one API call for all chunks
    → text_store.insert_source(meta)
    → vector_store.upsert_batch(chunks)    # write to Qdrant
    → text_store.insert_chunks_batch()     # write to SQLite
```

Key methods:
- `ingest_file(path, force=False)` — main entry point for files
- `ingest_url(url, force=False)` — main entry point for URLs
- `ingest_directory(dir_path)` — walks a directory recursively
- `delete_source(file_hash)` — removes from both Qdrant and SQLite
- `stats()` — returns source count, chunk count, vector point count
- `watch_directory(dir_path)` — uses [watchdog](https://python-watchdog.readthedocs.io/) to monitor a folder and auto-ingest on file creation/modification

---

### `ingestion/sources.py`

Extra data source connectors beyond what the basic file parsers handle.

**Notion** (`iter_notion_pages()`): walks your entire Notion workspace using the official API. Recursively extracts all block types — headings, paragraphs, bullet lists, to-dos, code blocks, etc. Requires `NOTION_TOKEN` env var. Get a token at [notion.so/my-integrations](https://www.notion.so/my-integrations).

**YouTube** (`parse_youtube(url)`): fetches public video transcripts using `youtube-transcript-api`. No API key needed — uses YouTube's public transcript endpoint. Timestamps are included in the text so you can find where in a video information came from. Accepts full URLs or bare video IDs.

**Obsidian** (`iter_obsidian_vault(vault_path)`): walks a vault directory, parses each `.md` file, and extracts Obsidian-specific metadata like `#tags` and backlinks.

---

### `storage/store.py`

Manages two databases simultaneously. Both get written in the same ingestion pass and read in the same retrieval pass.

**VectorStore (Qdrant)**

Stores embedding vectors with their payloads. The Qdrant client is a module-level singleton (`_QDRANT_CLIENT`) — all `VectorStore` instances share the same underlying connection. This prevents the "already accessed by another instance" error on Windows which only allows one process to hold a file lock at a time.

Uses `query_points()` for search (renamed from `search()` in qdrant-client v1.10+).

**TextStore (SQLite)**

Two tables:
- `sources` — one row per document (hash, path, type, title, dates)
- `chunks` — one row per chunk (chunk_id, file_hash, index, text)
- `chunks_fts` — FTS5 virtual table mirroring `chunks`, kept in sync via SQL triggers

The FTS5 virtual table enables BM25 keyword search directly in SQLite with no additional dependencies. SQL triggers on `INSERT` and `DELETE` keep it automatically in sync with the `chunks` table.

Chunk IDs are UUID5s derived from `file_hash:chunk_index` — deterministic, so re-ingesting the same document with `force=True` produces the same IDs.

---

### `retrieval/retriever.py`

The search engine. Combines three techniques:

**1. Multi-query expansion**

Before searching, the query is rewritten into 3 variants:
- Original query
- Keyword-only version (question words stripped)
- Noun-only version (stop words stripped)

Searching with multiple phrasings catches relevant chunks that might not match the exact wording of the original question.

**2. Hybrid search**

For each query variant, two searches run in parallel:
- Dense search in Qdrant (semantic similarity via cosine distance)
- Sparse search in SQLite FTS5 (BM25 keyword scoring)

Dense search finds semantically related content even when words don't match. Sparse search excels at exact technical terms, error messages, and proper nouns.

**3. Reciprocal Rank Fusion (RRF)**

Results from all searches are merged using RRF: each result gets a score of `1 / (rank + 60)`. Results appearing in multiple lists get their scores summed, effectively boosting documents that both methods agree on.

**4. Cross-encoder re-ranking**

After merging, the top candidates are re-scored using `cross-encoder/ms-marco-MiniLM-L6-v2`. Unlike embedding similarity (which compares vectors independently), a cross-encoder takes the query and each passage together as input and produces a single relevance score. Much more accurate but too slow to run on thousands of candidates — hence running it only on the merged shortlist.

Falls back to a TF-IDF heuristic if the cross-encoder model isn't available.

---

### `mcp_server/server.py`

Implements the [Model Context Protocol](https://modelcontextprotocol.io/) so Claude Desktop can use your Second Brain as a tool.

Exposes four tools:
- `search_brain(query, top_k?)` — runs the full retrieval pipeline and returns formatted results
- `add_document(path_or_url, force?)` — ingests a new source on the fly
- `list_sources()` — lists all ingested documents
- `delete_source(file_hash)` — removes a document

When connected to Claude Desktop, Claude can call these tools autonomously during a conversation — e.g. "let me check your notes on that" without you explicitly asking it to search.

---

### `interface/chat.py`

Terminal-based chat loop. Functionally equivalent to the web UI but runs in the console using [Rich](https://rich.readthedocs.io/) for formatting.

On each message:
1. Retrieves relevant chunks via `retriever.retrieve()`
2. Formats them as a numbered context block
3. Appends context + question to conversation history
4. Streams Claude's response token by token
5. Prints a source citation table

Maintains full conversation history so Claude can answer follow-up questions that reference earlier turns. History is capped at the last 20 messages to stay within context limits.

---

## Data flow example

When you ask "What is retrieval-augmented generation?":

```
1. retriever.py: embed("What is retrieval-augmented generation?")
   → [0.023, -0.14, 0.87, ...]   (384-dim vector)

2. vector_store.search(vector, top_k=10)
   → 10 chunks from Qdrant sorted by cosine similarity

3. text_store.fts_search('"retrieval" OR "augmented" OR "generation"', top_k=10)
   → 10 chunks from SQLite sorted by BM25 score

4. _merge(dense_results, sparse_results)
   → up to 20 unique chunks with RRF scores

5. cross_encoder.predict([(query, chunk_text) for each candidate])
   → re-ranked list, most relevant first

6. top 5 chunks injected into Claude prompt as:
   [1] Retrieval-augmented generation - Wikipedia (web)
   RAG is a technique that combines...

7. Claude streams response with citations [1][2][3]
```

---

## Extending the project

**Add a new file type parser:**
Add a function to `ingestion/parsers.py` following the same `(text, metadata)` return pattern, then add its extension to `EXTENSION_MAP`.

**Add a new data source:**
Add a function to `ingestion/sources.py` that yields `(text, metadata)` tuples, add a CLI command in `cli.py`, and add an endpoint in `api.py`.

**Improve retrieval quality:**
The cross-encoder in `retrieval/retriever.py` can be swapped for a larger model. Try `cross-encoder/ms-marco-MiniLM-L12-v2` for better accuracy at the cost of more RAM.

**Add authentication:**
The API has no auth. For exposing it on a network, add FastAPI's `HTTPBasic` or a bearer token check to the relevant endpoints.

**Scale up storage:**
Qdrant's local mode supports millions of vectors but uses file locks that prevent concurrent access. For multiple users or concurrent requests, run Qdrant as a standalone server (`docker run -p 6333:6333 qdrant/qdrant`) and change `config.py` to use `QDRANT_URL` instead of `QDRANT_PATH`.