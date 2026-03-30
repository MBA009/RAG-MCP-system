"""
api.py
───────
FastAPI backend for the Second Brain web UI.

Endpoints:
  POST /chat/stream     streaming chat (SSE)
  GET  /sources         list all ingested sources
  POST /ingest/url      ingest a URL on the fly
  POST /ingest/file     upload + ingest a file
  POST /ingest/youtube  ingest a YouTube video
  POST /ingest/notion   sync all Notion pages
  DELETE /source/{hash} delete a source
  GET  /stats           knowledge base stats

Run with:
  pip install fastapi uvicorn python-multipart
  uvicorn api:app --reload --port 8000

Then open: http://localhost:8000
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

import anthropic
from config          import ANTHROPIC_API_KEY, CLAUDE_MODEL, TOP_K_FINAL
from retrieval.retriever import Retriever
from ingestion.ingestor  import Ingestor
from storage.store       import TextStore

app = FastAPI(title="Second Brain API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Shared instances — initialized lazily on first request
_retriever  = None
_ingestor   = None
_text_store = None
ai_client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

def get_ingestor():
    global _ingestor
    if _ingestor is None:
        _ingestor = Ingestor()
    return _ingestor

def get_text_store():
    global _text_store
    if _text_store is None:
        _text_store = TextStore()
    return _text_store

SYSTEM_PROMPT = """You are a personal knowledge assistant with access to the user's Second Brain — 
a curated collection of their notes, PDFs, bookmarks, and documents.

Answer questions based primarily on the retrieved context provided.
Always cite your sources using [1], [2], etc. matching the passage numbers.
If the context doesn't cover the question, say so and share what you know from general knowledge.
Be concise but thorough."""


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []

class URLRequest(BaseModel):
    url: str

class YouTubeRequest(BaseModel):
    url: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "Second Brain API running", "docs": "/docs"}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        try:
            chunks = get_retriever().retrieve(req.message, top_k=TOP_K_FINAL)

            sources = [
                {
                    "title":       c.get("title", "Untitled"),
                    "source_path": c.get("source_path", ""),
                    "source_type": c.get("source_type", "?"),
                    "score":       round(float(c.get("score", 0)), 3),
                    "text":        c.get("text", "")[:300] + "…",
                }
                for c in chunks
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            context_lines = ["### Retrieved context:\n"]
            for i, c in enumerate(chunks, 1):
                context_lines.append(
                    f"[{i}] {c.get('title','Untitled')} ({c.get('source_type','?')})\n"
                    f"{c.get('text','')}\n"
                )
            context = "\n".join(context_lines)
            augmented = f"{context}\n### Question:\n{req.message}"

            messages = list(req.history)
            messages.append({"role": "user", "content": augmented})

            with ai_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    payload = json.dumps({"type": "token", "text": text})
                    yield f"data: {payload}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            import traceback
            print("CHAT ERROR:", traceback.format_exc())
            yield f"data: {json.dumps({'type': 'token', 'text': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/sources")
async def list_sources():
    return get_text_store().list_sources()


@app.get("/stats")
async def stats():
    return get_ingestor().stats()


@app.post("/ingest/url")
async def ingest_url(req: URLRequest):
    try:
        result = get_ingestor().ingest_url(req.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/youtube")
async def ingest_youtube(req: YouTubeRequest):
    try:
        from ingestion.sources import parse_youtube
        from ingestion.chunker  import split_into_chunks
        from ingestion.embedder import get_embedder
        from storage.store      import VectorStore, make_chunk_id

        text, meta = parse_youtube(req.url)
        result = get_ingestor()._ingest(text, meta, force=False)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/notion")
async def ingest_notion():
    try:
        from ingestion.sources import iter_notion_pages
        results = []
        for text, meta in iter_notion_pages():
            r = get_ingestor()._ingest(text, meta, force=False)
            results.append(r)
        ingested = sum(1 for r in results if r["status"] == "ingested")
        skipped  = sum(1 for r in results if r["status"] == "skipped")
        return {"ingested": ingested, "skipped": skipped, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        result = get_ingestor().ingest_file(tmp_path)
        result["original_name"] = file.filename
        tmp_path.unlink(missing_ok=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/source/{file_hash}")
async def delete_source(file_hash: str):
    get_ingestor().delete_source(file_hash)
    return {"deleted": file_hash}
