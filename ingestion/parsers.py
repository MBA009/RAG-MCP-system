"""
ingestion/parsers.py
────────────────────
Converts raw files → plain text + metadata dict.
Supports: PDF, DOCX, Markdown, HTML/URL, plain text, Python/code files.
"""
from __future__ import annotations
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


# ── Metadata helper ───────────────────────────────────────────────────────────

def _meta(path: Path, source_type: str, title: Optional[str] = None) -> dict:
    stat = path.stat()
    return {
        "source_path": str(path),
        "source_type": source_type,
        "title":       title or path.stem,
        "file_hash":   _sha256(path),
        "created_at":  datetime.utcfromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> tuple[str, dict]:
    import PyPDF2
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        title = (reader.metadata or {}).get("/Title") or path.stem
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts), _meta(path, "pdf", title)


def parse_docx(path: Path) -> tuple[str, dict]:
    from docx import Document
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs), _meta(path, "docx")


def parse_markdown(path: Path) -> tuple[str, dict]:
    import markdown
    raw = path.read_text(encoding="utf-8")
    # Strip front-matter (YAML between --- markers)
    raw = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL)
    # Convert to plain text (strip HTML tags)
    html  = markdown.markdown(raw)
    plain = re.sub(r"<[^>]+>", "", html)
    return plain.strip(), _meta(path, "markdown")


def parse_text(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, _meta(path, "text")


def parse_code(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lang = path.suffix.lstrip(".")
    meta = _meta(path, "code")
    meta["language"] = lang
    # Wrap in a comment header so the LLM knows what it's reading
    header = f"# File: {path.name}\n# Language: {lang}\n\n"
    return header + text, meta


def parse_url(url: str) -> tuple[str, dict]:
    """Fetch + parse a web page. Returns (text, metadata)."""
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(url, timeout=15, headers={"User-Agent": "SecondBrain/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove only true boilerplate — keep article body
    for tag in soup(["script", "style", "nav", "footer",
                     "header", "aside", "noscript", "iframe"]):
        tag.decompose()

    # For Wikipedia specifically, grab the article body directly
    main = (
        soup.find("div", {"id": "mw-content-text"}) or   # Wikipedia
        soup.find("main") or                               # Generic main tag
        soup.find("article") or                            # Article tag
        soup.find("div", {"id": "content"}) or             # Common CMS pattern
        soup.find("div", {"class": "content"}) or
        soup.body                                          # Fallback: whole body
    )

    title = soup.title.string.strip() if soup.title else urlparse(url).netloc
    text  = main.get_text(separator="\n", strip=True) if main else ""

    # Clean up excessive blank lines
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)

    meta = {
        "source_path": url,
        "source_type": "web",
        "title":       title,
        "file_hash":   hashlib.sha256(url.encode()).hexdigest(),
        "created_at":  datetime.utcnow().isoformat(),
        "modified_at": datetime.utcnow().isoformat(),
    }
    return text, meta

# ── Dispatcher ────────────────────────────────────────────────────────────────

EXTENSION_MAP = {
    ".pdf":   parse_pdf,
    ".docx":  parse_docx,
    ".md":    parse_markdown,
    ".txt":   parse_text,
    ".py":    parse_code,
    ".js":    parse_code,
    ".ts":    parse_code,
    ".go":    parse_code,
    ".rs":    parse_code,
    ".c":     parse_code,
    ".cpp":   parse_code,
    ".h":     parse_code,
    ".java":  parse_code,
}


def parse_file(path: Path) -> tuple[str, dict]:
    """Route a file to the correct parser. Raises ValueError for unknown types."""
    ext = path.suffix.lower()
    parser = EXTENSION_MAP.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser(path)
