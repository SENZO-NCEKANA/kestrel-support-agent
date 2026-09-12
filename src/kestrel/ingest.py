"""
Ingestion with content-hash incremental re-embedding.

Chunks whose content hash is unchanged are skipped. Chunks that vanished from
the corpus are deleted. Only genuinely new or edited text is sent to the
embedding API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_directory
from .embeddings import Embedder, get_embedder
from .store import Record, VectorStore, get_store


@dataclass
class IngestReport:
    total_chunks: int
    embedded: int
    skipped: int
    deleted: int
    provider: str

    def __str__(self) -> str:
        return (
            f"chunks={self.total_chunks} embedded={self.embedded} "
            f"unchanged={self.skipped} deleted={self.deleted} "
            f"provider={self.provider}"
        )


def ingest(kb_dir: str | Path, store: VectorStore, embedder: Embedder) -> IngestReport:
    chunks = chunk_directory(kb_dir)
    records = [
        Record(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            doc_title=c.doc_title,
            heading_path=c.heading_path,
            text=c.text,
            embed_text=c.embed_text,
            kind=c.kind,
            content_hash=c.content_hash,
            metadata=c.metadata,
        )
        for c in chunks
    ]

    existing = store.existing_hashes()
    current_ids = {r.chunk_id for r in records}

    to_embed = [r for r in records if existing.get(r.chunk_id) != r.content_hash]
    stale = [cid for cid in existing if cid not in current_ids]

    if to_embed:
        vectors = embedder.embed([r.embed_text for r in to_embed])
        store.upsert(to_embed, vectors)
    store.delete(stale)

    return IngestReport(
        total_chunks=len(records),
        embedded=len(to_embed),
        skipped=len(records) - len(to_embed),
        deleted=len(stale),
        provider=embedder.name,
    )


def build(kb_dir: str, db_path: str = "kestrel.db", provider: str | None = None):
    store = get_store("sqlite", path=db_path)
    embedder = get_embedder(provider)
    return ingest(kb_dir, store, embedder), store, embedder
