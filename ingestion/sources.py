"""
ingestion/sources.py
──────────────────────
Extra data source connectors:
  - Notion workspace (pages + databases)
  - YouTube video transcripts
  - Obsidian vault (already works via parse_markdown, but this adds metadata)
  - Any webpage (already in parsers.py, re-exported here for convenience)

Setup:
  Notion: pip install notion-client
          Set NOTION_TOKEN env var (get from notion.so/my-integrations)

  YouTube: pip install youtube-transcript-api
           No API key needed — uses public transcript endpoint
"""
from __future__ import annotations
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator


# ── Notion ────────────────────────────────────────────────────────────────────

def iter_notion_pages() -> Iterator[tuple[str, dict]]:
    """
    Yield (text, metadata) for every page in your Notion workspace.
    Requires: pip install notion-client
              NOTION_TOKEN environment variable
    """
    try:
        from notion_client import Client
    except ImportError:
        raise ImportError("Run: pip install notion-client")

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise ValueError("Set NOTION_TOKEN environment variable.\n"
                         "Get one at: https://www.notion.so/my-integrations")

    notion = Client(auth=token)

    # Search for all pages the integration can access
    cursor = None
    while True:
        kwargs = {"filter": {"property": "object", "value": "page"}, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp = notion.search(**kwargs)

        for page in resp.get("results", []):
            try:
                text, meta = _parse_notion_page(notion, page)
                if text.strip():
                    yield text, meta
            except Exception as e:
                print(f"  Skipping Notion page {page.get('id')}: {e}")

        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")


def _parse_notion_page(notion, page: dict) -> tuple[str, dict]:
    """Extract plain text from a Notion page's blocks."""
    page_id = page["id"]

    # Get title
    props = page.get("properties", {})
    title = ""
    for prop in props.values():
        if prop.get("type") == "title":
            parts = prop.get("title", [])
            title = "".join(p.get("plain_text", "") for p in parts)
            break
    if not title:
        title = f"Notion page {page_id[:8]}"

    # Get all blocks (recursive)
    text_parts = [title, ""]
    _extract_blocks(notion, page_id, text_parts)

    text = "\n".join(text_parts)
    url  = page.get("url", "")
    meta = {
        "source_path": url or f"notion:{page_id}",
        "source_type": "notion",
        "title":       title,
        "file_hash":   hashlib.sha256(page_id.encode()).hexdigest(),
        "created_at":  page.get("created_time", datetime.utcnow().isoformat()),
        "modified_at": page.get("last_edited_time", datetime.utcnow().isoformat()),
        "notion_id":   page_id,
    }
    return text, meta


def _extract_blocks(notion, block_id: str, parts: list, depth: int = 0):
    """Recursively walk blocks and append plain text."""
    if depth > 5:
        return  # prevent infinite recursion on deeply nested pages
    try:
        blocks = notion.blocks.children.list(block_id=block_id, page_size=100)
    except Exception:
        return

    for block in blocks.get("results", []):
        btype = block.get("type", "")
        content = block.get(btype, {})

        # Most block types have a "rich_text" array
        rich = content.get("rich_text", [])
        line = "".join(r.get("plain_text", "") for r in rich)

        if btype == "code":
            lang = content.get("language", "")
            parts.append(f"```{lang}\n{line}\n```")
        elif btype in ("heading_1", "heading_2", "heading_3"):
            parts.append(f"\n{'#' * int(btype[-1])} {line}")
        elif btype == "bulleted_list_item":
            parts.append(f"  • {line}")
        elif btype == "numbered_list_item":
            parts.append(f"  1. {line}")
        elif btype == "to_do":
            checked = "x" if content.get("checked") else " "
            parts.append(f"  [{checked}] {line}")
        elif btype == "divider":
            parts.append("---")
        elif line:
            parts.append(line)

        # Recurse into children
        if block.get("has_children"):
            _extract_blocks(notion, block["id"], parts, depth + 1)


# ── YouTube ───────────────────────────────────────────────────────────────────

def parse_youtube(url: str) -> tuple[str, dict]:
    """
    Fetch transcript from a YouTube video.
    Requires: pip install youtube-transcript-api
    Accepts: full URL or video ID
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
    except ImportError:
        raise ImportError("Run: pip install youtube-transcript-api")

    # Extract video ID from various URL formats
    video_id = _extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    # Try to get transcript (prefer English, fall back to any available)
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
            except Exception:
                transcript = next(iter(transcript_list))
        entries = transcript.fetch()
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        raise ValueError(f"No transcript available for {video_id}: {e}")

    # Build clean text with timestamps
    text_parts = []
    for entry in entries:
        start = int(entry["start"])
        mins, secs = divmod(start, 60)
        text_parts.append(f"[{mins:02d}:{secs:02d}] {entry['text']}")

    text = "\n".join(text_parts)

    # Try to get video title via oEmbed (no API key needed)
    title = _get_youtube_title(video_id) or f"YouTube: {video_id}"

    meta = {
        "source_path": f"https://www.youtube.com/watch?v={video_id}",
        "source_type": "youtube",
        "title":       title,
        "file_hash":   hashlib.sha256(video_id.encode()).hexdigest(),
        "created_at":  datetime.utcnow().isoformat(),
        "modified_at": datetime.utcnow().isoformat(),
        "video_id":    video_id,
    }
    return text, meta


def _extract_video_id(url: str) -> str | None:
    import re
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _get_youtube_title(video_id: str) -> str | None:
    try:
        import requests
        resp = requests.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=5
        )
        return resp.json().get("title")
    except Exception:
        return None


# ── Obsidian ──────────────────────────────────────────────────────────────────

def iter_obsidian_vault(vault_path: str) -> Iterator[tuple[str, dict]]:
    """
    Yield (text, metadata) for every .md file in an Obsidian vault.
    Adds Obsidian-specific metadata: tags, links, aliases from frontmatter.
    """
    import re
    from ingestion.parsers import parse_markdown

    vault = Path(vault_path)
    for md_file in vault.rglob("*.md"):
        if ".obsidian" in md_file.parts:
            continue  # skip Obsidian config files
        try:
            text, meta = parse_markdown(md_file)
            # Extract tags from content (Obsidian style: #tag)
            tags = re.findall(r'(?<!\S)#([A-Za-z][A-Za-z0-9_/-]*)', text)
            meta["tags"]   = list(set(tags))
            meta["vault"]  = str(vault)
            meta["source_type"] = "obsidian"
            if text.strip():
                yield text, meta
        except Exception as e:
            print(f"  Skipping {md_file.name}: {e}")
