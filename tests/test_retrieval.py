"""Retrieval and ingestion tests. Offline — uses the deterministic embedder."""
import tempfile
from pathlib import Path

import pytest

from kestrel.bm25 import BM25, tokenize
from kestrel.embeddings import HashEmbedder
from kestrel.ingest import ingest
from kestrel.retrieval import HybridRetriever
from kestrel.store import SQLiteStore

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
