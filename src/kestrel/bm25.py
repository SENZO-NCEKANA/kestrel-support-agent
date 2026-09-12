"""
BM25 Okapi, implemented directly rather than imported.

Dense retrieval alone loses on this corpus. A customer asking about "the R85
debit order fee" needs an exact lexical hit on R85; embeddings will happily
return the fee-schedule chunk about ATM charges because it is semantically
adjacent. Numbers, document codes and product names are exactly where dense
retrieval is weakest and BM25 is strongest, which is why the two are fused.

k1 controls term-frequency saturation, b controls length normalisation.
Defaults are the standard 1.5 / 0.75.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "i", "if", "in", "is", "it", "my", "of", "on", "or", "that", "the",
    "to", "was", "what", "when", "will", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip currency symbols and thousands separators.

    'R25 000' and 'R25000' must produce the same token stream, otherwise a
    customer's phrasing decides whether the fee waiver chunk is retrievable.

    R-prefixed amounts stay single tokens for free: lowercasing turns 'R85' into
    'r85' and _TOKEN's [a-z0-9]+ keeps it whole.
    """
    text = text.lower()
    text = re.sub(r"(?<=\d)[  ,](?=\d{3}\b)", "", text)  # 25 000 -> 25000
    tokens = _TOKEN.findall(text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(d) for d in corpus]
        self.n = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0
        self.tf = [Counter(d) for d in self.docs]

        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = [0.0] * self.n
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = 1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
        return scores

    def top_k(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
