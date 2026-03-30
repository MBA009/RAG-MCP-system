"""
storage/store.py
────────────────
Two complementary stores:

1. VectorStore  (Qdrant, local on-disk)
   – stores embeddings + payload (metadata + text snippet)
   – used for semantic / dense search

2. TextStore    (SQLite FTS5)
   – stores full chunk text + metadata
   – used for keyword / BM25 search and deduplication checks

Both share the same chunk_id (UUID derived from file_hash + chunk_index).
"""
from __future__ import annotations
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional
from config import SQLITE_PATH, QDRANT_PATH, COLLECTION, EMBED_DIM


# ── ID helper ─────────────────────────────────────────────────────────────────

def make_chunk_id(file_hash: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_hash}:{chunk_index}"))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vector Store (Qdrant)
# ─────────────────────────────────────────────────────────────────────────────

_QDRANT_CLIENT = None

def _get_qdrant_client():
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        _QDRANT_CLIENT = QdrantClient(path=str(QDRANT_PATH))
        existing = [c.name for c in _QDRANT_CLIENT.get_collections().collections]
        if COLLECTION not in existing:
            _QDRANT_CLIENT.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
    return _QDRANT_CLIENT


class VectorStore:
    def __init__(self):
        self.client = _get_qdrant_client()

    def upsert(self, chunk_id: str, vector: List[float], payload: dict):
        from qdrant_client.models import PointStruct
        self.client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=chunk_id, vector=vector, payload=payload)],
        )

    def upsert_batch(self, items: List[tuple]):
        from qdrant_client.models import PointStruct
        points = [PointStruct(id=cid, vector=vec, payload=pay) for cid, vec, pay in items]
        self.client.upsert(collection_name=COLLECTION, points=points)

    def search(self, query_vector: List[float], top_k: int = 10) -> List[dict]:
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {**r.payload, "score": r.score, "chunk_id": r.id}
            for r in results.points
        ]

    def delete_by_source(self, file_hash: str):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self.client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))]
            ),
        )

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION).count
# ─────────────────────────────────────────────────────────────────────────────
# 2. Text Store (SQLite + FTS5)
# ─────────────────────────────────────────────────────────────────────────────

class TextStore:
    def __init__(self):
        self.db_path = SQLITE_PATH
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    file_hash    TEXT PRIMARY KEY,
                    source_path  TEXT NOT NULL,
                    source_type  TEXT NOT NULL,
                    title        TEXT,
                    created_at   TEXT,
                    modified_at  TEXT,
                    ingested_at  TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id    TEXT PRIMARY KEY,
                    file_hash   TEXT NOT NULL REFERENCES sources(file_hash),
                    chunk_index INTEGER NOT NULL,
                    char_start  INTEGER,
                    text        TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    text,
                    title,
                    source_type,
                    content=chunks,
                    content_rowid=rowid
                );

                -- Keep FTS in sync with chunks table
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text, title, source_type)
                    VALUES (new.rowid, new.text,
                            (SELECT title FROM sources WHERE file_hash=new.file_hash),
                            (SELECT source_type FROM sources WHERE file_hash=new.file_hash));
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text, title, source_type)
                    VALUES ('delete', old.rowid, old.text,
                            (SELECT title FROM sources WHERE file_hash=old.file_hash),
                            (SELECT source_type FROM sources WHERE file_hash=old.file_hash));
                END;
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ── Write ─────────────────────────────────────────────────────────────────

    def insert_source(self, meta: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sources
                    (file_hash, source_path, source_type, title, created_at, modified_at)
                VALUES (:file_hash, :source_path, :source_type, :title, :created_at, :modified_at)
            """, meta)

    def insert_chunk(self, chunk_id: str, chunk: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chunks
                    (chunk_id, file_hash, chunk_index, char_start, text)
                VALUES (?, ?, ?, ?, ?)
            """, (chunk_id, chunk["file_hash"], chunk["chunk_index"],
                  chunk.get("char_start", 0), chunk["text"]))

    def insert_chunks_batch(self, items: List[tuple[str, dict]]):
        """items = list of (chunk_id, chunk_dict)"""
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks
                    (chunk_id, file_hash, chunk_index, char_start, text)
                VALUES (?, ?, ?, ?, ?)
            """, [
                (cid, c["file_hash"], c["chunk_index"], c.get("char_start", 0), c["text"])
                for cid, c in items
            ])

    # ── Read ──────────────────────────────────────────────────────────────────

    def source_exists(self, file_hash: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM sources WHERE file_hash=?", (file_hash,)
            ).fetchone()
        return row is not None

    def fts_search(self, query: str, top_k: int = 10) -> List[dict]:
        """Full-text BM25 search via FTS5."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT c.chunk_id, c.file_hash, c.chunk_index, c.text,
                       s.title, s.source_path, s.source_type,
                       rank
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.rowid
                JOIN sources s ON c.file_hash = s.file_hash
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, top_k)).fetchall()

        return [
            {
                "chunk_id":    r[0], "file_hash":   r[1],
                "chunk_index": r[2], "text":        r[3],
                "title":       r[4], "source_path": r[5],
                "source_type": r[6], "bm25_rank":   r[7],
            }
            for r in rows
        ]

    def list_sources(self) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT file_hash, source_path, source_type, title, ingested_at FROM sources"
            ).fetchall()
        return [
            {"file_hash": r[0], "source_path": r[1], "source_type": r[2],
             "title": r[3], "ingested_at": r[4]}
            for r in rows
        ]

    def delete_source(self, file_hash: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks  WHERE file_hash=?", (file_hash,))
            conn.execute("DELETE FROM sources WHERE file_hash=?", (file_hash,))

    def chunk_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
