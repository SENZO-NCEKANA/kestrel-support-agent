"""
Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion.

RRF rather than weighted score blending, because cosine similarity and BM25
scores live on incomparable scales. Normalising them requires a tuning
constant that silently rots as the corpus grows. RRF only uses rank position,
so it needs no calibration.

    score(d) = sum over retrievers of 1 / (k + rank(d))

k=60 is the value from the original Cormack et al. paper and is left alone
here deliberately: tuning it on 48 eval cases would be fitting noise.

Document precedence is deliberately NOT handled here. An earlier version
applied a blanket score multiplier to the escalation matrix so governance
rules could not lose a similarity contest to a fee table. It worked, and it
was wrong: it promoted KB-ESC-009 to rank 1 on every query, including a pure
limits question, displacing the correct answer.

Precedence is a routing decision, not a similarity one. The triage stage
knows when escalation rules are relevant; it passes `force_docs` and those
documents are guaranteed a slot. Queries with no escalation signal are left
alone. Encoding a governance rule as a score hack corrupts every unrelated
query, which is exactly what the eval caught.

Removing the boost exposed a second, deeper problem. The escalation matrix
still ranked first on a plain fee question, on lexical merit alone: it is a
meta-document that mentions every topic in the corpus ("fee and pricing
questions answered fully from the fee schedule"), so it competes on all of
them. Any governance document that enumerates what the others cover will do
this.

The fix is to stop treating one retrieval pool as serving two purposes.
Documents marked `customer_facing: false` are governance inputs for routing,
not sources for customer answers, and are excluded from the default pool
entirely. They enter only through `force_docs`, when triage asks for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .embeddings import Embedder, fingerprint
from .rerank import Reranker, get_reranker
from .store import EMBEDDER_KEY, Record, VectorStore

RRF_K = 60


@dataclass
class Hit:
    record: Record
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    # Set only when a reranker scored this hit. None under the default Noop
    # path, which is how the eval tells a reordered run from a plain one.
    rerank_score: float | None = None

    @property
    def doc_id(self) -> str:
        return self.record.doc_id

    def cite(self) -> str:
        path = f" > {self.record.heading_path}" if self.record.heading_path else ""
        return f"[{self.record.doc_id}] {self.record.doc_title}{path}"


def _is_internal(rec: Record) -> bool:
    return str(rec.metadata.get("customer_facing", "true")).lower() == "false"


class HybridRetriever:
    """Retrieves over customer-facing policy only.

    `records` is the answer pool. `all_records` includes internal governance
    documents, reachable solely via force_docs.
    """

    def __init__(self, store: VectorStore, embedder: Embedder,
                 reranker: Reranker | None = None):
        self.store = store
        self.embedder = embedder
        # Defaults to Noop unless RERANKER says otherwise, so every existing
        # caller keeps the behaviour it had.
        self.reranker = reranker or get_reranker()
        self.refresh()

    def refresh(self) -> None:
        self.all_records: list[Record] = self.store.all_records()
        self.records = [r for r in self.all_records if not _is_internal(r)]
        self.bm25 = BM25([r.embed_text for r in self.records]) if self.records else None
        # Cached with the records, so a retriever describes one consistent view
        # of the store until the next refresh. None on a store that predates
        # fingerprints; SQLiteStore.search catches that case on dimension.
        self.store_embedder: str | None = self.store.get_meta(EMBEDDER_KEY)

    def _check_embedding_space(self) -> None:
        """Refuse to score a query against vectors from another embedding space.

        Only called on the dense path. Lexical retrieval never embeds the query,
        so a mismatched embedder is harmless there and refusing it would break a
        mode that works.
        """
        query_space = fingerprint(self.embedder)
        if self.store_embedder and self.store_embedder != query_space:
            raise ValueError(
                f"The store was embedded with {self.store_embedder} but queries are "
                f"embedded with {query_space}. Vectors from different embedders "
                "cannot be compared. Re-ingest with this provider, or pass the "
                "--provider that built the store."
            )

    def retrieve(
        self,
        query: str,
        k: int = 6,
        candidates: int = 20,
        force_docs: set[str] | None = None,
        mode: str = "hybrid",
    ) -> list[Hit]:
        """Retrieve k chunks.

        force_docs guarantees at least one chunk from each named document,
        appended after the organically ranked hits. Used by triage to pull in
        governance rules without distorting relevance for everything else.

        mode selects which retrievers contribute to the fusion: "hybrid" uses
        both, "dense" and "lexical" isolate one. Isolating a single retriever
        leaves RRF a monotonic transform of that retriever's rank, so ordering
        collapses to it exactly — no separate code path is needed. The modes
        exist so the fusion can be measured against its own parts; a claim that
        hybrid beats BM25 alone is otherwise untestable.
        """
        if mode not in ("hybrid", "dense", "lexical"):
            raise ValueError(f"unknown retrieval mode: {mode}")
        if not self.records:
            return []

        dense: list[tuple[Record, float]] = []
        dense_rank: dict[str, int] = {}
        if mode != "lexical":
            self._check_embedding_space()
            qvec = self.embedder.embed([query])[0]
            dense = [
                (rec, score)
                for rec, score in self.store.search(qvec, candidates * 2)
                if not _is_internal(rec)
            ][:candidates]
            dense_rank = {rec.chunk_id: i for i, (rec, _) in enumerate(dense)}

        lexical_rank: dict[str, int] = {}
        if self.bm25 is not None and mode != "dense":
            for rank, (idx, _) in enumerate(self.bm25.top_k(query, candidates)):
                lexical_rank[self.records[idx].chunk_id] = rank

        by_id = {r.chunk_id: r for r in self.records}
        for rec, _ in dense:
            by_id.setdefault(rec.chunk_id, rec)

        fused: dict[str, float] = {}
        for chunk_id, rank in dense_rank.items():
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        for chunk_id, rank in lexical_rank.items():
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        def make_hit(cid: str, score: float) -> Hit:
            return Hit(
                record=by_id[cid],
                score=score,
                dense_rank=dense_rank.get(cid),
                lexical_rank=lexical_rank.get(cid),
            )

        # The reranker sees the whole candidate pool and decides what survives
        # it, rather than being handed a pre-truncated top-k it could only
        # reshuffle. Under NoopReranker this is exactly ranked[:k] as before —
        # which is what keeps the retrieval eval comparable across both paths.
        pool = [make_hit(cid, s) for cid, s in ranked[:candidates] if cid in by_id]
        hits = self.reranker.rerank(query, pool, k)

        if force_docs:
            present = {h.doc_id for h in hits}
            for doc_id in force_docs:
                if doc_id in present:
                    continue
                for cid, score in ranked:
                    rec = by_id.get(cid)
                    if rec and rec.doc_id == doc_id:
                        hits.append(make_hit(cid, score))
                        break
                else:
                    # Internal governance docs are outside the ranked pool.
                    for rec in self.all_records:
                        if rec.doc_id == doc_id:
                            hits.append(Hit(record=rec, score=0.0))
                            break
        return hits

    def retrieve_docs(self, query: str, k: int = 6, **kw) -> list[str]:
        """Distinct doc_ids in rank order. Used by the retrieval eval."""
        seen: list[str] = []
        for hit in self.retrieve(query, k=k, **kw):
            if hit.doc_id not in seen:
                seen.append(hit.doc_id)
        return seen


def format_context(hits: list[Hit], max_chars: int = 8000) -> str:
    """Render hits for the answer prompt, with citable headers."""
    parts, size = [], 0
    for hit in hits:
        block = f"--- {hit.cite()} ---\n{hit.record.text}"
        if size + len(block) > max_chars:
            break
        parts.append(block)
        size += len(block)
    return "\n\n".join(parts)
