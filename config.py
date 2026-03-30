"""
Central config — edit these to match your environment.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
WATCH_DIR    = DATA_DIR / "inbox"       # drop files here → auto-ingest
SQLITE_PATH  = DATA_DIR / "brain.db"
QDRANT_PATH  = DATA_DIR / "qdrant"      # local on-disk Qdrant (no server needed)

DATA_DIR.mkdir(exist_ok=True)
WATCH_DIR.mkdir(exist_ok=True)

# ── Embedding ─────────────────────────────────────────────────────────────────
# Option A: OpenAI  (set OPENAI_API_KEY env var)
# Option B: Local   (set EMBEDDING_BACKEND = "local")
EMBEDDING_BACKEND  = os.getenv("EMBEDDING_BACKEND", "openai")   # "openai" | "local"
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL    = "text-embedding-3-small"   # 1536-d, fast, cheap
LOCAL_EMBED_MODEL  = "all-MiniLM-L6-v2"         # 384-d, runs on CPU
EMBED_DIM          = 1536 if EMBEDDING_BACKEND == "openai" else 384

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512    # tokens (approx chars / 4)
CHUNK_OVERLAP = 64     # token overlap between adjacent chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE   = 10     # candidates from vector search
TOP_K_BM25    = 10     # candidates from full-text search
TOP_K_FINAL   = 5      # after re-ranking

# ── Claude ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-20250514"

# ── MCP Server ────────────────────────────────────────────────────────────────
MCP_SERVER_NAME = "second-brain"
MCP_PORT        = 8765

# ── Qdrant collection ─────────────────────────────────────────────────────────
COLLECTION = "brain"
