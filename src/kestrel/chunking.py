"""
Markdown-aware chunking for policy documents.

The whole point of this module is that policy knowledge lives in tables, and a
naive character-window chunker destroys them. Split the Kestrel limits table
between the verification-level rows and the tier-ceiling rows and no amount of
prompt engineering will recover the answer: the model is asked to compare two
numbers and only ever sees one.

Rules enforced here:

1. A table is atomic. It is never split unless it alone exceeds the hard max,
   in which case rows are split and the header row is repeated in each part.
2. A paragraph immediately preceding a table stays glued to it. That paragraph
   is almost always the sentence that says what the table means.
3. Every chunk carries the document title and its full heading path. A bare
   row of numbers is unretrievable; "Transaction Limits > Daily Limits by
   Verification Level" is not.
4. Frontmatter metadata (doc_id, version, owner) rides on every chunk so the
   answer layer can cite and the verifier can check precedence.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

TARGET_CHARS = 1200
MAX_CHARS = 2000
MIN_CHARS = 120

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


@dataclass
class Block:
    kind: str  # heading | table | paragraph | list | quote
    text: str
    level: int = 0


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    heading_path: str
    text: str
    chunk_index: int
    kind: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        digest = hashlib.sha256(
            f"{self.doc_id}:{self.chunk_index}:{self.text}".encode()
        ).hexdigest()[:16]
        return f"{self.doc_id}#{self.chunk_index}-{digest}"

    @property
    def embed_text(self) -> str:
        """Text actually sent to the embedder, with retrieval context prepended."""
        header = f"{self.doc_title}"
        if self.heading_path:
            header += f" > {self.heading_path}"
        return f"{header}\n\n{self.text}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embed_text.encode()).hexdigest()


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Minimal YAML frontmatter parser. Flat key: value pairs only."""
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("'\"")
    return meta, raw[match.end():]


def _split_blocks(body: str) -> list[Block]:
    """Group lines into semantic blocks, keeping table rows contiguous."""
    blocks: list[Block] = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            blocks.append(
                Block(kind="heading", text=heading.group(2).strip(), level=len(heading.group(1)))
            )
            i += 1
            continue

        if _TABLE_ROW.match(line):
            rows = []
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                rows.append(lines[i].rstrip())
                i += 1
            blocks.append(Block(kind="table", text="\n".join(rows)))
            continue

        if line.lstrip().startswith(">"):
            quote = []
            while i < len(lines) and (lines[i].lstrip().startswith(">") or lines[i].strip()):
                quote.append(lines[i].rstrip())
                i += 1
            blocks.append(Block(kind="quote", text="\n".join(quote)))
            continue

        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            items = []
            while i < len(lines) and lines[i].strip() and not _HEADING.match(lines[i]) \
                    and not _TABLE_ROW.match(lines[i]):
                items.append(lines[i].rstrip())
                i += 1
            blocks.append(Block(kind="list", text="\n".join(items)))
            continue

        para = []
        while i < len(lines) and lines[i].strip() and not _HEADING.match(lines[i]) \
                and not _TABLE_ROW.match(lines[i]) and not lines[i].lstrip().startswith(">"):
            para.append(lines[i].rstrip())
            i += 1
        blocks.append(Block(kind="paragraph", text="\n".join(para)))

    return blocks


def _split_large_table(table: str, max_chars: int) -> list[str]:
    """Split an oversized table by rows, repeating the header in every part."""
    rows = table.splitlines()
    if len(rows) < 3:
        return [table]

    header = rows[0]
    separator = rows[1] if _TABLE_SEP.match(rows[1]) else None
    body_start = 2 if separator else 1
    prefix = header + ("\n" + separator if separator else "")

    parts: list[str] = []
    current = [prefix]
    size = len(prefix)
    for row in rows[body_start:]:
        if size + len(row) + 1 > max_chars and len(current) > 1:
            parts.append("\n".join(current))
            current = [prefix, row]
            size = len(prefix) + len(row) + 1
        else:
            current.append(row)
            size += len(row) + 1
    if len(current) > 1:
        parts.append("\n".join(current))
    return parts or [table]


def chunk_document(raw: str, source_path: str = "") -> list[Chunk]:
    meta, body = parse_frontmatter(raw)
    doc_id = meta.get("doc_id") or Path(source_path).stem or "UNKNOWN"
    doc_title = meta.get("title") or doc_id

    blocks = _split_blocks(body)
    chunks: list[Chunk] = []
    heading_stack: list[str] = []

    buffer: list[Block] = []
    buffer_path = ""
    index = 0

    def current_path() -> str:
        # Drop the H1: it is the document title, already carried separately.
        # Leaving it in makes every citation read "Limits > Limits > Section 2".
        parts = [h for h in heading_stack[1:] if h]
        return " > ".join(parts)

    def flush():
        nonlocal buffer, index, buffer_path
        if not buffer:
            return
        text = "\n\n".join(b.text for b in buffer).strip()
        if not text:
            buffer = []
            return
        kind = "table" if any(b.kind == "table" for b in buffer) else "prose"
        chunks.append(
            Chunk(
                doc_id=doc_id,
                doc_title=doc_title,
                heading_path=buffer_path,
                text=text,
                chunk_index=index,
                kind=kind,
                metadata=meta,
            )
        )
        index += 1
        buffer = []

    def buffer_size() -> int:
        return sum(len(b.text) + 2 for b in buffer)

    for pos, block in enumerate(blocks):
        if block.kind == "heading":
            flush()
            heading_stack = heading_stack[: block.level - 1]
            while len(heading_stack) < block.level - 1:
                heading_stack.append("")
            heading_stack.append(block.text)
            buffer_path = current_path()
            continue

        if not buffer:
            buffer_path = current_path()

        if block.kind == "table" and len(block.text) > MAX_CHARS:
            # Oversized table: emit each part as its own chunk, header repeated.
            trailing_para = buffer[-1] if buffer and buffer[-1].kind == "paragraph" else None
            if trailing_para is not None:
                buffer = buffer[:-1]
            flush()
            for part in _split_large_table(block.text, MAX_CHARS):
                if trailing_para is not None:
                    buffer = [trailing_para, Block(kind="table", text=part)]
                else:
                    buffer = [Block(kind="table", text=part)]
                buffer_path = current_path()
                flush()
            continue

        # Would adding this block overflow the target?
        if buffer and buffer_size() + len(block.text) > TARGET_CHARS:
            if block.kind == "table" and buffer[-1].kind == "paragraph" \
                    and buffer_size() + len(block.text) <= MAX_CHARS:
                # Keep the explanatory paragraph glued to its table.
                buffer.append(block)
                continue
            if block.kind == "table" and buffer[-1].kind == "paragraph":
                # Move that paragraph forward with the table into a new chunk.
                lead = buffer.pop()
                flush()
                buffer = [lead, block]
                buffer_path = current_path()
                continue
            flush()
            buffer_path = current_path()

        buffer.append(block)

        if buffer_size() >= TARGET_CHARS and block.kind != "paragraph":
            flush()

    flush()

    # Merge any runt chunks forward so single stray sentences don't become
    # their own embedding.
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk.text) < MIN_CHARS
            and merged[-1].heading_path == chunk.heading_path
            and len(merged[-1].text) + len(chunk.text) < MAX_CHARS
        ):
            merged[-1].text += "\n\n" + chunk.text
        else:
            merged.append(chunk)

    for n, chunk in enumerate(merged):
        chunk.chunk_index = n
    return merged


def chunk_directory(kb_dir: str | Path) -> list[Chunk]:
    kb_path = Path(kb_dir)
    out: list[Chunk] = []
    for path in sorted(kb_path.glob("*.md")):
        out.extend(chunk_document(path.read_text(), str(path)))
    return out
