"""
retrieval/retriever.py  (v2 — upgraded)
────────────────────────────────────────
Hybrid retrieval: dense (Qdrant) + sparse (BM25/FTS5) → cross-encoder re-rank → top-K.

Upgrades from v1:
  - Real ML cross-encoder (ms-marco-MiniLM-L6-v2) replaces heuristic scoring
  - Multi-query expansion: rewrites question 3 ways, merges all result sets
  - Better dedup and score normalisation
"""
from __future__ import annotations
import math
import re
from typing import List
from config import TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL

_CROSS_ENCODER = None

def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
            print("Cross-encoder loaded")
        except Exception as e:
            print(f"Cross-encoder unavailable ({e}), using fallback")
            _CROSS_ENCODER = "fallback"
    return _CROSS_ENCODER


class Retriever:
    def __init__(self):
        from storage.store      import VectorStore, TextStore
        from ingestion.embedder import get_embedder
        self.vector_store = VectorStore()
        self.text_store   = TextStore()
        self.embedder     = get_embedder()

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> List[dict]:
        queries = _expand_query(query)
        all_dense:  List[dict] = []
        all_sparse: List[dict] = []
        for q in queries:
            q_vec = self.embedder.embed(q)
            all_dense.extend(self.vector_store.search(q_vec, top_k=TOP_K_DENSE))
            fts_q = _sanitize_fts(q)
            all_sparse.extend(self.text_store.fts_search(fts_q, top_k=TOP_K_BM25))
        candidates = _merge(all_dense, all_sparse)
        return _rerank(query, candidates)[:top_k]


def _expand_query(query: str) -> List[str]:
    q = query.strip()
    variants = [q]
    keyword_version = re.sub(
        r'^(what|how|why|when|where|who|which|can|could|should|is|are|does|do)\s+',
        '', q, flags=re.IGNORECASE).strip()
    if keyword_version and keyword_version != q:
        variants.append(keyword_version)
    stops = {'the','a','an','of','in','on','at','to','for','with','about',
              'from','by','as','is','are','was','were','be','been','being',
              'have','has','had','do','does','did','will','would','could',
              'should','may','might','must','can','me','my','i','you','your',
              'it','its','this','that','these','those'}
    nouns = [w for w in re.findall(r'\w+', q.lower()) if w not in stops and len(w) > 2]
    if nouns and len(nouns) < len(q.split()):
        variants.append(' '.join(nouns))
    return list(dict.fromkeys(variants))


def _sanitize_fts(query: str) -> str:
    words = re.findall(r'\w+', query)
    if not words:
        return '""'
    return " OR ".join(f'"{w}"' for w in words)


def _merge(dense: List[dict], sparse: List[dict]) -> List[dict]:
    seen: dict[str, dict] = {}
    for rank, item in enumerate(dense):
        cid = str(item.get("chunk_id", ""))
        if not cid:
            continue
        rrf = 1.0 / (rank + 60)
        if cid in seen:
            seen[cid]["_rrf"] += rrf
        else:
            item["_rrf"] = rrf
            seen[cid] = item
    for rank, item in enumerate(sparse):
        cid = str(item.get("chunk_id", ""))
        if not cid:
            continue
        rrf = 1.0 / (rank + 60)
        if cid in seen:
            seen[cid]["_rrf"] += rrf
        else:
            item["_rrf"] = rrf
            seen[cid] = item
    return list(seen.values())


def _rerank(query: str, candidates: List[dict]) -> List[dict]:
    if not candidates:
        return candidates
    encoder = _get_cross_encoder()
    if encoder != "fallback":
        pairs  = [(query, c.get("text", "")) for c in candidates]
        scores = encoder.predict(pairs)
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
    else:
        q_tokens = set(re.findall(r'\w+', query.lower()))
        for item in candidates:
            text     = (item.get("text") or "").lower()
            t_tokens = set(re.findall(r'\w+', text))
            if q_tokens and t_tokens:
                overlap = len(q_tokens & t_tokens)
                tfidf   = overlap / math.sqrt(len(q_tokens) * len(t_tokens))
            else:
                tfidf = 0.0
            item["score"] = 0.7 * item.get("_rrf", 0) + 0.3 * tfidf
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates
