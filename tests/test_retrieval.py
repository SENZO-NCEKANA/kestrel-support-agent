"""Retrieval and ingestion tests. Offline — uses the deterministic embedder."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kestrel.bm25 import BM25, tokenize
from kestrel.embeddings import HashEmbedder
from kestrel.ingest import ingest
from kestrel.retrieval import HybridRetriever
from kestrel.store import EMBEDDER_KEY, SQLiteStore

KB = Path(__file__).resolve().parents[1] / "kb"


@pytest.fixture(scope="module")
def retriever():
    tmp = tempfile.mkdtemp()
    store = SQLiteStore(str(Path(tmp) / "test.db"))
    embedder = HashEmbedder()
    ingest(KB, store, embedder)
    return HybridRetriever(store, embedder)


def test_currency_tokens_normalise():
    """'R25 000' and 'R25000' must tokenise identically."""
    assert tokenize("R25 000") == tokenize("R25000")
    assert "r25000" in tokenize("balance of R25 000 or more")


def test_bm25_ranks_exact_numeric_match():
    corpus = [
        "ATM withdrawal fees are charged per transaction",
        "The unpaid debit order fee is R85 on all tiers",
        "Card replacement costs vary by account tier",
    ]
    bm = BM25(corpus)
    top = bm.top_k("R85 debit order fee", k=1)
    assert top and top[0][0] == 1


def test_ingest_is_incremental():
    tmp = tempfile.mkdtemp()
    db = str(Path(tmp) / "inc.db")
    store = SQLiteStore(db)
    embedder = HashEmbedder()

    first = ingest(KB, store, embedder)
    assert first.embedded == first.total_chunks
    assert first.skipped == 0

    second = ingest(KB, store, embedder)
    assert second.embedded == 0, "unchanged corpus should re-embed nothing"
    assert second.skipped == second.total_chunks
    assert second.deleted == 0


def test_ingest_deletes_removed_chunks():
    tmp = Path(tempfile.mkdtemp())
    kb = tmp / "kb"
    kb.mkdir()
    doc = "---\ndoc_id: KB-TMP-001\ntitle: Temp\nversion: 1\n---\n\n# One\n\n" + ("word " * 200)
    (kb / "a.md").write_text(doc)

    store = SQLiteStore(str(tmp / "d.db"))
    embedder = HashEmbedder()
    first = ingest(kb, store, embedder)
    assert first.total_chunks >= 1

    (kb / "a.md").unlink()
    second = ingest(kb, store, embedder)
    assert second.deleted == first.total_chunks
    assert second.total_chunks == 0


# ------------------------------------------------------------ embedding space
#
# HashEmbedder(dim=256) and HashEmbedder(dim=128) have different fingerprints,
# which is enough to exercise every path here without a network call.

def _fresh_store(name: str = "space.db") -> SQLiteStore:
    return SQLiteStore(str(Path(tempfile.mkdtemp()) / name))


class _FailingEmbedder(HashEmbedder):
    name = "failing"

    def embed(self, texts):
        raise RuntimeError("embedding API unavailable")


def test_switching_embedder_reembeds_everything():
    """Regression: the re-embed decision used the content hash alone.

    Switching to OpenAI embeddings changed no hash, so nothing was re-embedded,
    the report named a provider it had not used, and the first query crashed on
    a dimension mismatch far from the cause.
    """
    store = _fresh_store()
    first = ingest(KB, store, HashEmbedder(dim=256))
    switched = ingest(KB, store, HashEmbedder(dim=128))

    assert switched.embedded == switched.total_chunks == first.total_chunks
    assert "embedder changed hash-fake::256 -> hash-fake::128" in str(switched)
    assert HybridRetriever(store, HashEmbedder(dim=128)).retrieve("monthly fee on Plus")


def test_new_embedder_is_recorded_so_the_next_run_skips():
    store = _fresh_store()
    ingest(KB, store, HashEmbedder(dim=256))
    ingest(KB, store, HashEmbedder(dim=128))
    again = ingest(KB, store, HashEmbedder(dim=128))
    assert again.embedded == 0
    assert again.note == ""


def test_store_without_a_fingerprint_is_fully_reembedded():
    """A database built before fingerprints cannot say which space it is in."""
    store = _fresh_store()
    ingest(KB, store, HashEmbedder())
    store.conn.execute("DELETE FROM meta")
    store.conn.commit()

    report = ingest(KB, store, HashEmbedder())
    assert report.embedded == report.total_chunks
    assert "predates embedder fingerprints" in report.note


def test_failed_embedding_leaves_the_old_fingerprint():
    """The fingerprint must describe the vectors actually stored."""
    store = _fresh_store()
    ingest(KB, store, HashEmbedder())
    before = store.get_meta(EMBEDDER_KEY)

    with pytest.raises(RuntimeError):
        ingest(KB, store, _FailingEmbedder(dim=128))
    assert store.get_meta(EMBEDDER_KEY) == before


def test_full_reembed_still_removes_deleted_documents():
    tmp = Path(tempfile.mkdtemp())
    kb = tmp / "kb"
    kb.mkdir()
    head = "---\ndoc_id: KB-TMP-00{n}\ntitle: Temp {n}\nversion: 1\n---\n\n# Section\n\n"
    (kb / "a.md").write_text(head.format(n=1) + "alpha " * 200)
    (kb / "b.md").write_text(head.format(n=2) + "beta " * 200)

    store = SQLiteStore(str(tmp / "s.db"))
    ingest(kb, store, HashEmbedder(dim=256))
    (kb / "b.md").unlink()
    second = ingest(kb, store, HashEmbedder(dim=128))

    assert second.deleted > 0
    assert {r.doc_id for r in store.all_records()} == {"KB-TMP-001"}


def test_retriever_refuses_a_query_from_another_embedding_space():
    store = _fresh_store()
    ingest(KB, store, HashEmbedder(dim=256))
    retriever = HybridRetriever(store, HashEmbedder(dim=128))

    with pytest.raises(ValueError) as exc:
        retriever.retrieve("monthly fee on Plus")
    assert "hash-fake::256" in str(exc.value)
    assert "hash-fake::128" in str(exc.value)


def test_lexical_mode_never_embeds_so_it_ignores_the_space():
    store = _fresh_store()
    ingest(KB, store, HashEmbedder(dim=256))
    retriever = HybridRetriever(store, HashEmbedder(dim=128))
    assert retriever.retrieve("monthly fee on Plus", mode="lexical")


def test_store_search_names_a_dimension_mismatch():
    """Legacy stores have no fingerprint, so the store itself must refuse."""
    store = _fresh_store()
    ingest(KB, store, HashEmbedder(dim=256))
    with pytest.raises(ValueError, match="dimensions"):
        store.search(np.zeros(128, dtype=np.float32), k=3)


def test_retriever_returns_hits(retriever):
    hits = retriever.retrieve("What is the monthly fee on the Plus account?", k=5)
    assert hits
    assert all(h.record.doc_id for h in hits)


def test_force_docs_guarantees_inclusion(retriever):
    """Triage can pull in governance rules explicitly."""
    hits = retriever.retrieve(
        "what is the monthly fee on the Plus account", k=4,
        force_docs={"KB-ESC-009"},
    )
    assert "KB-ESC-009" in {h.doc_id for h in hits}


def test_no_blanket_precedence_on_unrelated_query(retriever):
    """Regression: the escalation matrix must not top-rank a pure fee query.

    An earlier score-multiplier version of precedence promoted KB-ESC-009 to
    rank 1 on every query. This asserts it stays out unless asked for.
    """
    hits = retriever.retrieve(
        "how much is the monthly account fee on Plus and how do I avoid it", k=5
    )
    assert hits[0].doc_id != "KB-ESC-009"


def test_scenario_one_retrieves_both_limit_tables(retriever):
    """Demo scenario 1 needs the verification table AND the tier table.

    Comparing R2 000 against R10 000 is impossible if only one is retrieved,
    no matter how good the answer prompt is.
    """
    hits = retriever.retrieve(
        "I am a Kestrel Private customer and I can only draw R2 000 at the ATM. "
        "Your website says Private is R10 000. Please fix this.",
        k=6,
    )
    text = "\n".join(h.record.text for h in hits)
    assert "Daily Limits by Verification Level" in " ".join(h.record.heading_path for h in hits), \
        "verification-level limits table not retrieved"
    assert "Tier Ceilings" in " ".join(h.record.heading_path for h in hits), \
        "tier ceilings table not retrieved"


def test_hits_are_citable(retriever):
    hits = retriever.retrieve("dispute window for goods not received", k=3)
    for h in hits:
        assert h.cite().startswith("[KB-")


def test_empty_store_returns_nothing():
    tmp = tempfile.mkdtemp()
    store = SQLiteStore(str(Path(tmp) / "empty.db"))
    r = HybridRetriever(store, HashEmbedder())
    assert r.retrieve("anything") == []


def test_retrieval_modes_are_distinct(retriever):
    """The --mode flag must actually change retrieval.

    Regression: mode was a CLI argument that only ever reached the header line
    of the eval output. All three settings printed different labels and ran the
    full hybrid, so benchmarking fusion against its parts silently compared a
    thing to itself. Asserting the rankings differ keeps it wired.
    """
    q = "what does a replacement card cost after confirmed fraud on my account"
    hybrid = [h.record.chunk_id for h in retriever.retrieve(q, k=6, mode="hybrid")]
    dense = [h.record.chunk_id for h in retriever.retrieve(q, k=6, mode="dense")]
    lexical = [h.record.chunk_id for h in retriever.retrieve(q, k=6, mode="lexical")]

    assert hybrid and dense and lexical
    assert dense != lexical, "dense and lexical returned identical rankings"


def test_unknown_mode_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("anything", mode="bm25")
