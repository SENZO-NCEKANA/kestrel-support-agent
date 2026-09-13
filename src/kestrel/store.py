"""
Vector store behind a protocol, with two backends.

SQLiteStore runs with zero infrastructure so the project is clonable and
runnable in one command. PgVectorStore is the production path. Both satisfy
the same interface, so swapping is a config change rather than a rewrite —
which is the actual point of putting a protocol here.

Incremental ingestion: every chunk carries a content hash. Re-ingesting an
unchanged corpus costs nothing. Editing one policy document re-embeds only
the chunks that changed. On a real KB this is the difference between a
cent and a rand every time someone fixes a typo.

A hash only means "unchanged" inside one embedding space, so a store also
records the fingerprint of the embedder that built it, under `EMBEDDER_KEY`.
The invariant is store-wide rather than per chunk: vectors from two embedders
are not comparable even at equal dimension, so a store holds exactly one space
and switching is all-or-nothing.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

# Store metadata key holding kestrel.embeddings.fingerprint() of the embedder
# that wrote the vectors.
EMBEDDER_KEY = "embedder"


@dataclass
class Record:
    chunk_id: str
    doc_id: str
    doc_title: str
    heading_path: str
    text: str
    embed_text: str
    kind: str
    content_hash: str
    metadata: dict


class VectorStore(Protocol):
    def upsert(self, records: list[Record], vectors: np.ndarray) -> None: ...
    def existing_hashes(self) -> dict[str, str]: ...
    def delete(self, chunk_ids: list[str]) -> None: ...
    def all_records(self) -> list[Record]: ...
    def search(self, vector: np.ndarray, k: int) -> list[tuple[Record, float]]: ...
    def get_meta(self, key: str) -> str | None: ...
    def set_meta(self, key: str, value: str) -> None: ...


class SQLiteStore:
    """Vectors stored as float32 blobs; brute-force cosine over a small corpus.

    Exact search over a few hundred chunks is microseconds and avoids an ANN
    index that would only matter at a scale this corpus never reaches.
    """

    def __init__(self, path: str = "kestrel.db"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                doc_id       TEXT NOT NULL,
                doc_title    TEXT NOT NULL,
                heading_path TEXT,
                text         TEXT NOT NULL,
                embed_text   TEXT NOT NULL,
                kind         TEXT,
                content_hash TEXT NOT NULL,
                metadata     TEXT,
                vector       BLOB NOT NULL,
                dim          INTEGER NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_id)")
        # IF NOT EXISTS, so a database built before fingerprints gains the table
        # on open with no migration step. It starts empty, which ingest reads as
        # "space unknown" and answers by re-embedding.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()

    def upsert(self, records: list[Record], vectors: np.ndarray) -> None:
        rows = []
        for rec, vec in zip(records, vectors):
            v = np.asarray(vec, dtype=np.float32)
            rows.append(
                (
                    rec.chunk_id, rec.doc_id, rec.doc_title, rec.heading_path,
                    rec.text, rec.embed_text, rec.kind, rec.content_hash,
                    json.dumps(rec.metadata), v.tobytes(), len(v),
                )
            )
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        self.conn.commit()

    def existing_hashes(self) -> dict[str, str]:
        cur = self.conn.execute("SELECT chunk_id, content_hash FROM chunks")
        return dict(cur.fetchall())

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self.conn.executemany(
            "DELETE FROM chunks WHERE chunk_id = ?", [(c,) for c in chunk_ids]
        )
        self.conn.commit()

    def _rows(self):
        return self.conn.execute(
            "SELECT chunk_id, doc_id, doc_title, heading_path, text, embed_text,"
            " kind, content_hash, metadata, vector, dim FROM chunks"
        ).fetchall()

    @staticmethod
    def _to_record(row) -> Record:
        return Record(
            chunk_id=row[0], doc_id=row[1], doc_title=row[2], heading_path=row[3],
            text=row[4], embed_text=row[5], kind=row[6], content_hash=row[7],
            metadata=json.loads(row[8] or "{}"),
        )

    def all_records(self) -> list[Record]:
        return [self._to_record(r) for r in self._rows()]

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Record, float]]:
        rows = self._rows()
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r[9], dtype=np.float32) for r in rows])
        q = np.asarray(vector, dtype=np.float32)

        # The retriever checks fingerprints first; this is the last line of
        # defence for a store built before fingerprints existed. Without it the
        # failure is numpy's matmul error, which names a gufunc signature and
        # says nothing about embedders.
        if mat.shape[1] != q.shape[0]:
            raise ValueError(
                f"query vector has {q.shape[0]} dimensions but {self.path} holds "
                f"{mat.shape[1]}-dimensional vectors. The store was built by a "
                "different embedder than the one embedding this query: re-ingest "
                "with this provider, or query with the one that built it."
            )

        qn = np.linalg.norm(q)
        if qn:
            q = q / qn
        sims = mat @ q
        order = np.argsort(-sims)[:k]
        return [(self._to_record(rows[i]), float(sims[i])) for i in order]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


class PgVectorStore:
    """Production backend. Same interface, pgvector for ANN at scale.

    Requires: CREATE EXTENSION vector; and psycopg installed.

    The `vector({dim})` column is fixed when the table is created. The embedder
    fingerprint makes a switch visible to ingest, but a switch to a different
    dimension still needs a fresh table here — pgvector will reject the upsert
    rather than store it, which is the right failure, not a silent one.

    Untested in this repository: CI has no Postgres, and the meta methods below
    mirror the SQLite ones rather than having been run.
    """

    def __init__(self, dsn: str, dim: int = 1536):
        import psycopg  # lazy import

        self.conn = psycopg.connect(dsn)
        self.dim = dim
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id     TEXT PRIMARY KEY,
                    doc_id       TEXT NOT NULL,
                    doc_title    TEXT NOT NULL,
                    heading_path TEXT,
                    text         TEXT NOT NULL,
                    embed_text   TEXT NOT NULL,
                    kind         TEXT,
                    content_hash TEXT NOT NULL,
                    metadata     JSONB,
                    vector       vector({dim}) NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_vec ON chunks "
                "USING hnsw (vector vector_cosine_ops)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
        self.conn.commit()

    def existing_hashes(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, content_hash FROM chunks")
            return dict(cur.fetchall())

    def get_meta(self, key: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT value FROM meta WHERE key = %s", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO meta (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, value),
            )
        self.conn.commit()

    def upsert(self, records: list[Record], vectors: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            for rec, vec in zip(records, vectors):
                cur.execute(
                    """
                    INSERT INTO chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embed_text = EXCLUDED.embed_text,
                        content_hash = EXCLUDED.content_hash,
                        vector = EXCLUDED.vector
                    """,
                    (
                        rec.chunk_id, rec.doc_id, rec.doc_title, rec.heading_path,
                        rec.text, rec.embed_text, rec.kind, rec.content_hash,
                        json.dumps(rec.metadata), list(map(float, vec)),
                    ),
                )
        self.conn.commit()

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
        self.conn.commit()

    def _record(self, row) -> Record:
        return Record(
            chunk_id=row[0], doc_id=row[1], doc_title=row[2], heading_path=row[3],
            text=row[4], embed_text=row[5], kind=row[6], content_hash=row[7],
            metadata=row[8] or {},
        )

    def all_records(self) -> list[Record]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, doc_id, doc_title, heading_path, text,"
                " embed_text, kind, content_hash, metadata FROM chunks"
            )
            return [self._record(r) for r in cur.fetchall()]

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Record, float]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, doc_id, doc_title, heading_path, text,"
                " embed_text, kind, content_hash, metadata,"
                " 1 - (vector <=> %s::vector) AS score"
                " FROM chunks ORDER BY vector <=> %s::vector LIMIT %s",
                (list(map(float, vector)), list(map(float, vector)), k),
            )
            return [(self._record(r), float(r[9])) for r in cur.fetchall()]


def get_store(backend: str = "sqlite", **kwargs) -> VectorStore:
    if backend == "sqlite":
        return SQLiteStore(kwargs.get("path", "kestrel.db"))
    if backend == "pgvector":
        return PgVectorStore(kwargs["dsn"], kwargs.get("dim", 1536))
    raise ValueError(f"unknown store backend: {backend}")
