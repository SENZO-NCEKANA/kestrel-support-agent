"""
Pluggable embedding providers.

Two implementations behind one protocol:

- OpenAIEmbedder  — production
- HashEmbedder    — deterministic, offline, no API key

The hash embedder exists so the test suite and CI can exercise the full
retrieval path without credentials or spend. It is a random projection of a
bag of words: lexically sensitive, semantically blind. Useful for asserting
that plumbing works. Never use it to make a claim about retrieval quality.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np

from .bm25 import tokenize


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic offline embedder. Tests and CI only."""

    name = "hash-fake"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        from openai import OpenAI  # imported lazily so the package is optional

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it, or use "
                "EMBEDDING_PROVIDER=hash for the offline path."
            )
        self.client = OpenAI(api_key=key)
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        arr = np.array(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)


def get_embedder(provider: str | None = None) -> Embedder:
    provider = provider or os.environ.get("EMBEDDING_PROVIDER", "hash")
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embedding provider: {provider}")
