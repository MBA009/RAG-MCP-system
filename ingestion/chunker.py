"""
ingestion/chunker.py
────────────────────
Splits long text into overlapping chunks.
Uses a simple character-based sliding window (works without a tokeniser).
Each chunk carries its parent metadata + positional info.
"""
from __future__ import annotations
from typing import Iterator
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Approximate chars per token (good enough for splitting purposes)
CHARS_PER_TOKEN = 4
CHUNK_CHARS    = CHUNK_SIZE    * CHARS_PER_TOKEN   # ≈ 2048 chars
OVERLAP_CHARS  = CHUNK_OVERLAP * CHARS_PER_TOKEN   # ≈ 256 chars


def split_into_chunks(text: str, metadata: dict) -> Iterator[dict]:
    """
    Yield dicts with keys:
        text        – the chunk content
        chunk_index – position in document (0-based)
        char_start  – character offset in original text
        **metadata  – all parent metadata fields
    """
    text = text.strip()
    if not text:
        return

    # ── Sentence-aware split ─────────────────────────────────────────────────
    # Try to break at sentence boundaries rather than mid-word.
    sentences = _split_sentences(text)
    chunks    = _pack_sentences(sentences)

    for idx, (start, chunk_text) in enumerate(chunks):
        yield {
            **metadata,
            "text":        chunk_text,
            "chunk_index": idx,
            "char_start":  start,
        }


def _split_sentences(text: str) -> list[tuple[int, str]]:
    """Return (char_offset, sentence_text) pairs."""
    import re
    # Split on '. ', '! ', '? ', '\n\n', keeping the delimiter
    pattern = r'(?<=[.!?])\s+|(?<=\n)\n+'
    parts   = re.split(pattern, text)
    result  = []
    pos     = 0
    for part in parts:
        part = part.strip()
        if part:
            result.append((pos, part))
        pos += len(part) + 1   # +1 for the space/newline
    return result


def _pack_sentences(sentences: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Greedily fill chunks up to CHUNK_CHARS chars.
    Add OVERLAP_CHARS of the previous chunk to the start of the next.
    """
    chunks: list[tuple[int, str]] = []
    current_parts: list[str] = []
    current_len:   int       = 0
    current_start: int       = 0

    for offset, sent in sentences:
        sent_len = len(sent)

        # If this single sentence is already huge, hard-split it
        if sent_len > CHUNK_CHARS:
            if current_parts:
                chunks.append((current_start, " ".join(current_parts)))
                current_parts, current_len = [], 0
            for i in range(0, sent_len, CHUNK_CHARS - OVERLAP_CHARS):
                chunks.append((offset + i, sent[i : i + CHUNK_CHARS]))
            current_start = offset + sent_len
            continue

        if current_len + sent_len > CHUNK_CHARS and current_parts:
            chunks.append((current_start, " ".join(current_parts)))
            # Overlap: keep last N chars of previous chunk
            overlap_text  = " ".join(current_parts)[-OVERLAP_CHARS:]
            current_parts = [overlap_text, sent]
            current_len   = len(overlap_text) + sent_len
            current_start = offset
        else:
            if not current_parts:
                current_start = offset
            current_parts.append(sent)
            current_len += sent_len + 1

    if current_parts:
        chunks.append((current_start, " ".join(current_parts)))

    return chunks
