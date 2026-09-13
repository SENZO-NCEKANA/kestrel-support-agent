"""
Ingestion with content-hash incremental re-embedding.

Chunks whose content hash is unchanged are skipped. Chunks that vanished from
the corpus are deleted. Only genuinely new or edited text is sent to the
embedding API.

A content hash only means "unchanged" inside one embedding space. The first
version keyed the skip on the hash alone, so switching from the offline hash
embedder to OpenAI re-embedded nothing: every hash still matched, the report
printed `embedded=0 provider=openai`, and the store kept its 256-dimensional
vectors until the first query died in a numpy matmul error nowhere near the
cause. Each store now records the fingerprint of the embedder that built it,
and a different fingerprint — or none, on a store that predates this — means
everything is re-embedded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_directory
from .embeddings import Embedder, fingerprint, get_embedder
from .store import EMBEDDER_KEY, Record, VectorStore, get_store


@dataclass
class IngestReport:
    total_chunks: int
    embedded: int
    skipped: int
    deleted: int
    provider: str
    embedder: str = ""
    # Why a full re-embed happened, when one did. Empty on an ordinary run.
    note: str = ""

    def __str__(self) -> str:
        line = (
            f"chunks={self.total_chunks} embedded={self.embedded} "
            f"unchanged={self.skipped} deleted={self.deleted} "
            f"provider={self.provider}"
        )
        return f"{line}\n{self.note}" if self.note else line


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

    current = fingerprint(embedder)
    previous = store.get_meta(EMBEDDER_KEY)
    stored = store.existing_hashes()
    current_ids = {r.chunk_id for r in records}

    note = ""
    if previous == current:
        to_embed = [r for r in records if stored.get(r.chunk_id) != r.content_hash]
    else:
        # A different space, or an unmarked store whose space cannot be known.
        # Guessing is the one thing that must not happen here: re-embedding costs
        # nothing offline and a fraction of a cent against OpenAI.
        to_embed = records
        if previous is not None:
            note = f"embedder changed {previous} -> {current}; re-embedded all"
        elif stored:
            note = f"store predates embedder fingerprints; re-embedded all as {current}"

    # From the real stored hashes, so deleted documents are removed even on a
    # full re-embed.
    stale = [cid for cid in stored if cid not in current_ids]

    if to_embed:
        vectors = embedder.embed([r.embed_text for r in to_embed])
        store.upsert(to_embed, vectors)
    store.delete(stale)

    # Recorded last, and only once the vectors are written. If embedding raises —
    # a 429, a timeout — the store keeps its old fingerprint and still describes
    # the vectors it actually holds.
    store.set_meta(EMBEDDER_KEY, current)

    return IngestReport(
        total_chunks=len(records),
        embedded=len(to_embed),
        skipped=len(records) - len(to_embed),
        deleted=len(stale),
        provider=embedder.name,
        embedder=current,
        note=note,
    )


def build(kb_dir: str, db_path: str = "kestrel.db", provider: str | None = None):
    store = get_store("sqlite", path=db_path)
    embedder = get_embedder(provider)
    return ingest(kb_dir, store, embedder), store, embedder
