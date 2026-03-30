"""
ingestion/ingestor.py
──────────────────────
Orchestrates the full pipeline for a single document:
  parse → chunk → embed → store (Qdrant + SQLite)

Also exposes:
  - ingest_file(path)
  - ingest_url(url)
  - ingest_directory(dir_path)
  - watch_directory(dir_path)   ← uses watchdog to auto-ingest new files
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parsers  import parse_file, parse_url
from ingestion.chunker  import split_into_chunks
from ingestion.embedder import get_embedder
from storage.store      import VectorStore, TextStore, make_chunk_id
from config             import WATCH_DIR


class Ingestor:
    def __init__(self):
        self.vector_store = VectorStore()
        self.text_store   = TextStore()
        self.embedder     = get_embedder()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_file(self, path: Path, force: bool = False) -> dict:
        """
        Ingest a single file. Returns a status dict.
        Set force=True to re-ingest even if already present.
        """
        path = Path(path)
        text, meta = parse_file(path)
        return self._ingest(text, meta, force)

    def ingest_url(self, url: str, force: bool = False) -> dict:
        text, meta = parse_url(url)
        return self._ingest(text, meta, force)

    def ingest_directory(self, dir_path: Path, force: bool = False) -> list[dict]:
        results = []
        for file in Path(dir_path).rglob("*"):
            if file.is_file():
                try:
                    result = self.ingest_file(file, force=force)
                    results.append(result)
                    print(f"  ✓ {file.name} — {result['chunks']} chunks")
                except ValueError as e:
                    print(f"  ✗ {file.name} — skipped: {e}")
                except Exception as e:
                    print(f"  ✗ {file.name} — error: {e}")
        return results

    def delete_source(self, file_hash: str):
        self.vector_store.delete_by_source(file_hash)
        self.text_store.delete_source(file_hash)

    def stats(self) -> dict:
        return {
            "total_chunks":  self.text_store.chunk_count(),
            "vector_points": self.vector_store.count(),
            "sources":       len(self.text_store.list_sources()),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ingest(self, text: str, meta: dict, force: bool) -> dict:
        file_hash = meta["file_hash"]

        if not force and self.text_store.source_exists(file_hash):
            return {"status": "skipped", "title": meta.get("title"), "chunks": 0}

        # Delete existing version if re-ingesting
        if force and self.text_store.source_exists(file_hash):
            self.delete_source(file_hash)

        # 1. Chunk
        chunks = list(split_into_chunks(text, meta))
        if not chunks:
            return {"status": "empty", "title": meta.get("title"), "chunks": 0}

        # 2. Embed (batch for efficiency)
        texts    = [c["text"] for c in chunks]
        vectors  = self.embedder.embed_batch(texts)

        # 3. Store source
        self.text_store.insert_source(meta)

        # 4. Store chunks (Qdrant + SQLite in one pass)
        qdrant_batch   = []
        sqlite_batch   = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = make_chunk_id(file_hash, chunk["chunk_index"])
            payload  = {
                "text":        chunk["text"],
                "title":       chunk.get("title", ""),
                "source_path": chunk.get("source_path", ""),
                "source_type": chunk.get("source_type", ""),
                "file_hash":   file_hash,
                "chunk_index": chunk["chunk_index"],
            }
            qdrant_batch.append((chunk_id, vector, payload))
            sqlite_batch.append((chunk_id, chunk))

        self.vector_store.upsert_batch(qdrant_batch)
        self.text_store.insert_chunks_batch(sqlite_batch)

        return {
            "status": "ingested",
            "title":  meta.get("title"),
            "chunks": len(chunks),
            "hash":   file_hash,
        }


# ── Directory watcher ─────────────────────────────────────────────────────────

def watch_directory(dir_path: Path = WATCH_DIR):
    """
    Block and watch a directory. Any new/modified file gets auto-ingested.
    Run this in a background thread or separate process.
    """
    from watchdog.observers import Observer
    from watchdog.events    import FileSystemEventHandler

    ingestor = Ingestor()

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                path = Path(event.src_path)
                print(f"\n📥 New file detected: {path.name}")
                try:
                    result = ingestor.ingest_file(path)
                    print(f"   ✓ {result['chunks']} chunks ingested")
                except Exception as e:
                    print(f"   ✗ {e}")

        def on_modified(self, event):
            if not event.is_directory:
                path = Path(event.src_path)
                print(f"\n🔄 File modified: {path.name}")
                try:
                    result = ingestor.ingest_file(path, force=True)
                    print(f"   ✓ Re-ingested {result['chunks']} chunks")
                except Exception as e:
                    print(f"   ✗ {e}")

    observer = Observer()
    observer.schedule(Handler(), str(dir_path), recursive=True)
    observer.start()
    print(f"👁️  Watching {dir_path} for new files…")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
