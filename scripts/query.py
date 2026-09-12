#!/usr/bin/env python3
"""
Ad-hoc retrieval inspector. Ask the knowledge base anything and see what comes
back, why it ranked, and what the answer layer would actually be handed.

    python3 scripts/query.py "what does a replacement card cost after fraud"
    python3 scripts/query.py --force KB-ESC-009 "monthly fee on Plus"
    python3 scripts/query.py --context "dispute window goods not received"

Exists because recall@6 is an aggregate. When a number looks wrong the only way
to find out why is to look at the chunks.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kestrel.embeddings import get_embedder
from kestrel.rerank import get_reranker
from kestrel.retrieval import HybridRetriever, format_context
from kestrel.store import get_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--db", default="kestrel.db")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--force", action="append", default=[],
                    help="doc_id triage would force in, e.g. KB-ESC-009")
    ap.add_argument("--context", action="store_true",
                    help="print the context block the answer prompt would receive")
    ap.add_argument("--reranker", default=None,
                    help="noop (default) | cross-encoder")
    args = ap.parse_args()

    store = get_store("sqlite", path=args.db)
    retriever = HybridRetriever(store, get_embedder(args.provider),
                                get_reranker(args.reranker))
    if not retriever.records:
        print("No chunks in store. Run scripts/ingest.py first.")
        sys.exit(1)

    query = " ".join(args.query)
    hits = retriever.retrieve(query, k=args.k, force_docs=set(args.force) or None)

    print(f"\nquery: {query}")
    print(f"pool: {len(retriever.records)} answerable chunks "
          f"({len(retriever.all_records) - len(retriever.records)} internal, excluded)\n")

    for n, h in enumerate(hits, 1):
        d = "—" if h.dense_rank is None else f"{h.dense_rank + 1}"
        l = "—" if h.lexical_rank is None else f"{h.lexical_rank + 1}"
        forced = "  [forced]" if h.score == 0.0 else ""
        print(f"{n}. {h.cite()}{forced}")
        print(f"   rrf={h.score:.5f}  dense_rank={d}  bm25_rank={l}  kind={h.record.kind}")
        print(f"   {h.record.text[:110].replace(chr(10), ' ')}...\n")

    if args.context:
        print("--- context handed to the answer prompt ---")
        print(format_context(hits))


if __name__ == "__main__":
    main()
