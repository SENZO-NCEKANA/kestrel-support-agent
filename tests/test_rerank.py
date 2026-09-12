"""
Reranker tests.

Split deliberately. The Noop tests are the ones that matter to CI: they pin the
property that the default path is byte-identical to having no reranker at all,
which is what lets the retrieval eval compare the two honestly and what keeps
the build free of a model download.

The cross-encoder tests skip unless onnxruntime is installed. They assert
ordering behaviour, not a recall number — a recall claim belongs in the eval,
run against the whole set, not in a unit test against two handpicked strings.
"""
import tempfile
from pathlib import Path

import pytest

from kestrel.embeddings import HashEmbedder
from kestrel.ingest import ingest
from kestrel.rerank import NoopReranker, get_reranker
from kestrel.retrieval import HybridRetriever
from kestrel.store import SQLiteStore

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "kb"

HAVE_ONNX = True
try:  # pragma: no cover - environment dependent
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401
except ImportError:  # pragma: no cover
    HAVE_ONNX = False


@pytest.fixture(scope="module")
def store_and_embedder():
    tmp = tempfile.mkdtemp()
    store = SQLiteStore(str(Path(tmp) / "rerank.db"))
    embedder = HashEmbedder()
    ingest(KB, store, embedder)
    return store, embedder


# ------------------------------------------------------------------------ noop

def test_default_is_noop():
    """CI must never reach for a model download by accident."""
    assert get_reranker().name == "noop"


def test_unknown_reranker_raises():
    with pytest.raises(ValueError):
        get_reranker("magic")


def test_noop_preserves_fused_order_and_truncates(store_and_embedder):
    """The identity property the retrieval eval's comparability rests on."""
    store, embedder = store_and_embedder
    r = HybridRetriever(store, embedder, NoopReranker())
    q = "what does a replacement card cost after confirmed fraud"
    pool = r.retrieve(q, k=20, candidates=20)
    top = r.retrieve(q, k=6, candidates=20)
    assert [h.record.chunk_id for h in top] == [h.record.chunk_id for h in pool[:6]]


def test_noop_leaves_rerank_score_unset(store_and_embedder):
    """How the eval tells a reordered run from a plain one."""
    store, embedder = store_and_embedder
    r = HybridRetriever(store, embedder, NoopReranker())
    hits = r.retrieve("monthly fee on Plus", k=6)
    assert all(h.rerank_score is None for h in hits)


def test_noop_handles_an_empty_pool():
    assert NoopReranker().rerank("q", [], 6) == []


# ------------------------------------------------------------- cross-encoder

pytestmark_ce = pytest.mark.skipif(
    not HAVE_ONNX, reason="onnxruntime/tokenizers not installed")


@pytestmark_ce
def test_cross_encoder_prefers_the_semantically_right_chunk(store_and_embedder):
    """RS-01's failure mode, in isolation.

    'Should I put it in a Fixed Deposit or move it offshore, what would you do'
    shares no vocabulary with the complaints document that says support cannot
    advise and names who can. Lexical retrieval cannot bridge that; this is the
    one thing the reranker is here to do.
    """
    from kestrel.rerank import CrossEncoderReranker

    store, embedder = store_and_embedder
    r = HybridRetriever(store, embedder, CrossEncoderReranker())
    docs = r.retrieve_docs(
        "Where should I put my bonus\nI am getting a R400 000 bonus. Should I "
        "put it in a Kestrel Fixed Deposit or move it offshore? What would you do?",
        k=6)
    assert "KB-CMP-008" in docs


@pytestmark_ce
def test_cross_encoder_sets_scores_in_descending_order(store_and_embedder):
    from kestrel.rerank import CrossEncoderReranker

    store, embedder = store_and_embedder
    r = HybridRetriever(store, embedder, CrossEncoderReranker())
    hits = r.retrieve("dispute window for goods never delivered", k=6)
    scores = [h.rerank_score for h in hits]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)


@pytestmark_ce
def test_cross_encoder_handles_an_empty_pool():
    from kestrel.rerank import CrossEncoderReranker

    assert CrossEncoderReranker().rerank("q", [], 6) == []
