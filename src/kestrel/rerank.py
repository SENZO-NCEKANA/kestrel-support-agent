"""
Pluggable rerankers, mirroring the embeddings and llm modules exactly.

- CrossEncoderReranker — a real cross-encoder, local, no API key
- NoopReranker         — returns the fused order untouched. Default.

Retrieval and reranking answer different questions. A bi-encoder embeds the
query and the document separately and compares two vectors that never met; it
has to be that way, because the document vectors are computed once at ingest
and reused for every query. A cross-encoder embeds the pair *together* and
attends across both, so it can register that "my chargeback was declined, what
are my options" and "escalating a declined dispute to the ombud" are the same
question. That is strictly more expressive and strictly too slow to run over a
corpus — which is the whole reason for the two-stage shape: fuse cheaply to
~20 candidates, then score those properly.

Why this exists here specifically. The retrieval eval has two standing misses,
both KB-CMP-008, on KM-06 and RS-01:

    KM-06  "My chargeback was declined and I do not accept it.
            What are my options now?"
    RS-01  "Should I put it in a Fixed Deposit or move it offshore?
            What would you do?"

Neither ticket uses a single word from the document that answers it —
"complaint", "ombud", "referral". The customer describes a situation; the policy
names a process. BM25 cannot bridge that, and the offline hash embedder is a
random projection of a bag of words, so it cannot either. Before this module,
KB-CMP-008 ranked 7th of 8 documents on KM-06 and 6th of 8 on RS-01: present in
the candidate pool, buried beneath documents that merely share vocabulary.

That is the gap a cross-encoder is for, and it is the only remaining miss in the
set — which makes it a fair test rather than a feature looking for a use.

Scale caveat, stated up front: the customer-facing corpus is 33 chunks across 8
documents. A 20-candidate pool is therefore about 60% of everything there is,
and reranking it is a much easier problem than reranking 20 of 200 000. The
architecture is the production one; the numbers it produces here are not
production numbers.

The default stays Noop so CI and the offline path keep costing nothing and
needing no model download.

Why ONNX rather than sentence-transformers. The obvious implementation is
`sentence_transformers.CrossEncoder`, which is three lines. It cannot be used
here: sentence-transformers requires torch, and torch publishes no macOS
x86_64 wheel for Python 3.13 — `pip install torch` on this machine resolves to
nothing at all, on any version. That is a platform fact, not a version to pin
around.

onnxruntime runs the same MiniLM cross-encoder as an exported graph, needs no
torch, and installs in ~50MB against torch's ~2GB. `tokenizers` does the
WordPiece encoding that `transformers` would otherwise pull torch in to do.
The trade is that the pair encoding and the sigmoid are written out by hand
below instead of being hidden inside `.predict()` — about fifteen lines, and
they are the fifteen lines worth understanding anyway.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # avoids a circular import at runtime; retrieval imports this
    from .retrieval import Hit

# Xenova's ONNX export of cross-encoder/ms-marco-MiniLM-L-6-v2 — the standard
# baseline for this task, same weights, no torch needed to run it. Not tuned on
# the eval set: tuning a reranker on 48 cases would be fitting noise, the same
# argument that keeps RRF_K at the paper's value.
DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MAX_TOKENS = 512  # the model's limit; policy chunks sit well under it


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list["Hit"], k: int) -> list["Hit"]: ...


class NoopReranker:
    """Keeps the fused order. The default, and the CI path.

    Truncating to k here rather than upstream means the retriever always builds
    the same candidate pool and the reranker decides what survives it. Under
    Noop that is identical to the pre-reranker behaviour, which is what makes
    the retrieval eval comparable across both.
    """

    name = "noop"

    def rerank(self, query: str, hits: list["Hit"], k: int) -> list["Hit"]:
        return hits[:k]


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly and reorders by that score."""

    name = "cross-encoder"

    def __init__(self, model: str | None = None):
        # Lazy, so the packages stay optional exactly as openai does for
        # OpenAILLM. Nothing on the default path imports any of this.
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "RERANKER=cross-encoder needs onnxruntime, tokenizers and "
                "huggingface-hub. Install with `pip3 install onnxruntime "
                "tokenizers huggingface-hub`, or unset RERANKER for the "
                "offline path."
            ) from exc

        self.model_name = model or os.environ.get("RERANK_MODEL", DEFAULT_MODEL)

        # Downloaded once and cached in ~/.cache/huggingface thereafter.
        self.session = ort.InferenceSession(
            hf_hub_download(self.model_name, "onnx/model.onnx")
        )
        self._inputs = {i.name for i in self.session.get_inputs()}

        self.tokenizer = Tokenizer.from_file(
            hf_hub_download(self.model_name, "tokenizer.json")
        )
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(MAX_TOKENS)

    def _score(self, query: str, texts: list[str]) -> list[float]:
        """Joint (query, chunk) relevance logits, one per text."""
        import numpy as np

        encoded = self.tokenizer.encode_batch([(query, t) for t in texts])
        feed = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded],
                                       dtype=np.int64),
        }
        # BERT-family exports want segment ids; some exports omit them.
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in encoded],
                                              dtype=np.int64)

        # Shape (batch, 1). Raw logits, left unsquashed: only the ordering is
        # used, and a sigmoid is monotonic, so it would change the numbers
        # printed and nothing about the ranking.
        return [float(row[0]) for row in self.session.run(None, feed)[0]]

    def rerank(self, query: str, hits: list["Hit"], k: int) -> list["Hit"]:
        if not hits:
            return []

        scores = self._score(query, [hit.record.text for hit in hits])

        # The cross-encoder score replaces the RRF score rather than blending
        # with it. Blending would need a weight, and a weight needs calibration
        # against something — the same objection that put RRF in retrieval.py
        # instead of weighted score fusion. The reranker either knows better
        # than the fusion or it does not.
        ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]),
                        reverse=True)
        out = []
        for hit, score in ranked[:k]:
            hit.rerank_score = float(score)
            out.append(hit)
        return out


def get_reranker(provider: str | None = None) -> Reranker:
    provider = provider or os.environ.get("RERANKER", "noop")
    if provider == "cross-encoder":
        return CrossEncoderReranker()
    if provider == "noop":
        return NoopReranker()
    raise ValueError(f"unknown reranker: {provider}")
