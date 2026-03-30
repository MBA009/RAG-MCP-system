"""
ingestion/embedder.py
──────────────────────
Converts text → dense float vectors.
Backend A: OpenAI text-embedding-3-small (API call, best quality)
Backend B: sentence-transformers all-MiniLM-L6-v2 (local, no cost)

Usage:
    embedder = get_embedder()
    vec = embedder.embed("Hello world")          # single
    vecs = embedder.embed_batch(["Hello", "World"]) # batch
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from config import EMBEDDING_BACKEND, OPENAI_API_KEY, EMBEDDING_MODEL, LOCAL_EMBED_MODEL


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]: ...
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model  = EMBEDDING_MODEL

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=text.replace("\n", " "),
        )
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t.replace("\n", " ") for t in texts]
        # OpenAI supports up to 2048 inputs per call
        results = []
        for i in range(0, len(cleaned), 256):
            batch = cleaned[i:i+256]
            resp  = self.client.embeddings.create(model=self.model, input=batch)
            results.extend([r.embedding for r in resp.data])
        return results


class LocalEmbedder(BaseEmbedder):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(LOCAL_EMBED_MODEL)

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=64).tolist()


_EMBEDDER: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    """Singleton — instantiate once, reuse everywhere."""
    global _EMBEDDER
    if _EMBEDDER is None:
        if EMBEDDING_BACKEND == "openai":
            _EMBEDDER = OpenAIEmbedder()
        else:
            _EMBEDDER = LocalEmbedder()
    return _EMBEDDER
