#!/usr/bin/env python3
"""
Index Kestral policies into Qdrant for dense retrieval.
Each chunk is a semantic section of a policy document.
The embedded text includes the policy name and section path for better retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# Configuration (environment overrides)
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kestral_policies")
DENSE_URL = os.getenv("DENSE_URL", "http://localhost:8200").rstrip("/")
DENSE_DIM = int(os.getenv("DENSE_DIM", "384"))
BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "32")))
RETRIES = max(1, int(os.getenv("RETRIES", "3")))
TIMEOUT = float(os.getenv("TIMEOUT", "30.0"))

MARKDOWN_EXTS = {".md", ".markdown", ".mdown"}

# ---------------------------------------------------------------------------
# Markdown → semantic chunks
# ---------------------------------------------------------------------------

def parse_policy_sections(text: str) -> list[dict[str, Any]]:
    """
    Split a markdown document into sections based on ATX headings.
    Each section becomes a chunk with:
      - heading_path: list of headings from H1 to current (e.g. ["Returns & Refunds", "Electronics", "Conditions"])
      - content: all text under that heading (excluding the heading itself)
      - level: heading level (2 for ##, 3 for ###, etc.)
    We skip H1-only chunks (they're just the title).
    """
    lines = text.splitlines(keepends=True)
    # pattern for ATX headings: optional spaces, #s, space, title
    heading_re = re.compile(r"^(#{1,6})\s+(.*)")

    sections: list[dict[str, Any]] = []
    current_heading_path: list[str] = []
    current_level = 0
    current_lines: list[str] = []

    def flush_section():
        if current_lines:
            content = "".join(current_lines).strip()
            if content:
                sections.append({
                    "heading_path": current_heading_path.copy(),
                    "content": content,
                    "level": current_level,
                })
            current_lines.clear()

    for line in lines:
        m = heading_re.match(line)
        if m:
            # flush previous section
            flush_section()

            hashes = m.group(1)
            title = m.group(2).strip()
            level = len(hashes)

            # adjust heading_path
            if level == 1:
                # top-level: reset path
                current_heading_path = [title]
            else:
                # trim deeper levels and replace current
                current_heading_path = current_heading_path[:level-1]
                current_heading_path.append(title)

            current_level = level
            continue

        # regular line
        current_lines.append(line)

    flush_section()
    return sections


def extract_policy_name(text: str) -> str:
    """Return the first H1 title, or 'Untitled'."""
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    return m.group(1).strip() if m else "Untitled Policy"


def build_chunks(md_text: str, file_path: Path) -> list[dict[str, Any]]:
    """Convert markdown text into chunks ready for indexing."""
    policy_name = extract_policy_name(md_text)
    sections = parse_policy_sections(md_text)

    chunks: list[dict[str, Any]] = []
    for i, sec in enumerate(sections):
        # Skip chunks that are just boilerplate or extremely short
        if len(sec["content"]) < 20:
            continue

        heading_path_str = " > ".join(sec["heading_path"])
        # This is what we embed – includes context
        text_to_embed = f"Policy: {policy_name}\nSection: {heading_path_str}\n{sec['content']}"

        chunks.append({
            "source_path": str(file_path),
            "chunk_index": i,
            "policy_name": policy_name,
            "section_title": sec["heading_path"][-1] if sec["heading_path"] else "",
            "heading_path": heading_path_str,
            "content": sec["content"],
            "text_to_embed": text_to_embed,
            "tags": extract_tags(sec["content"]),
        })

    return chunks


def extract_tags(content: str) -> list[str]:
    """Derive simple tags from content keywords."""
    tags = set()
    low = content.lower()
    if any(w in low for w in ("refund", "return", "money back")):
        tags.add("refunds")
    if any(w in low for w in ("damage", "defect", "doa", "broken")):
        tags.add("damaged_items")
    if any(w in low for w in ("delivery", "shipping", "courier", "pin code")):
        tags.add("delivery")
    if any(w in low for w in ("warranty", "repair", "service center", "protect")):
        tags.add("warranty")
    if any(w in low for w in ("cancel", "cancellation")):
        tags.add("cancellation")
    if any(w in low for w in ("payment", "cod", "upi", "card", "wallet")):
        tags.add("payments")
    if any(w in low for w in ("complaint", "escalat", "grievance", "nodal")):
        tags.add("escalation")
    return sorted(tags)


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def make_point_id(source_path: str, chunk_index: int) -> str:
    """Deterministic UUIDv5 from file path + chunk index."""
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, source_path)
    return str(uuid.uuid5(namespace, str(chunk_index)))


def embed_texts(http: httpx.Client, texts: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for _ in range(RETRIES):
        try:
            resp = http.post(f"{DENSE_URL}/embed", json={"texts": texts})
            resp.raise_for_status()
            data = resp.json()
            # support both 'vectors' and 'embeddings'
            vectors = data.get("vectors") or data.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                raise RuntimeError("dense service returned invalid vectors")
            return [[float(x) for x in vec] for vec in vectors]
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"embedding failed: {last_error}")


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create collection if needed; optionally recreate it."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
        else:
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=DENSE_DIM,
            distance=models.Distance.COSINE,
        ),
    )


# ---------------------------------------------------------------------------
# Main indexing logic
# ---------------------------------------------------------------------------

def index_policies(input_path: str, recreate: bool = False) -> int:
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Gather markdown files
    if root.is_file():
        files = [root]
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in MARKDOWN_EXTS)

    # Parse all chunks
    all_chunks: list[dict[str, Any]] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = build_chunks(text, file_path)
        all_chunks.extend(chunks)

    if not all_chunks:
        print(json.dumps({"collection": COLLECTION_NAME, "indexed": 0}))
        return 0

    # Connect to Qdrant
    qdrant_kwargs: dict[str, Any] = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        qdrant_kwargs["api_key"] = QDRANT_API_KEY
    client = QdrantClient(**qdrant_kwargs)

    ensure_collection(client, recreate=recreate)

    # Embed and upsert in batches
    with httpx.Client(timeout=TIMEOUT) as http:
        for offset in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[offset : offset + BATCH_SIZE]
            texts = [c["text_to_embed"] for c in batch]
            vectors = embed_texts(http, texts)

            points = []
            for chunk, vector in zip(batch, vectors):
                pid = make_point_id(chunk["source_path"], chunk["chunk_index"])
                payload = {
                    "text": chunk["content"],  # original section content
                    "embedded_text": chunk["text_to_embed"],  # what was actually embedded
                    "source_path": chunk["source_path"],
                    "chunk_index": chunk["chunk_index"],
                    "policy_name": chunk["policy_name"],
                    "section_title": chunk["section_title"],
                    "heading_path": chunk["heading_path"],
                    "tags": chunk["tags"],
                }
                points.append(
                    models.PointStruct(
                        id=pid,
                        vector=vector,
                        payload=payload,
                    )
                )

            client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(json.dumps({
        "collection": COLLECTION_NAME,
        "indexed": len(all_chunks),
        "files": len(files),
    }))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index Kestral policy markdown files into Qdrant"
    )
    parser.add_argument(
        "input",
        help="Path to a .md file or directory containing .md files",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing collection and re-index from scratch",
    )
    args = parser.parse_args()

    return index_policies(args.input, recreate=args.recreate)


if __name__ == "__main__":
    raise SystemExit(main())