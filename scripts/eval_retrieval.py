#!/usr/bin/env python3
"""
Retrieval evaluation: recall@k against expected_sources in the eval set.

This measures only the retrieval stage. If recall@k is poor, no prompt will
save the answer — the model is being asked about text it never received.
Isolating this before touching generation is the whole reason it exists as a
separate runner.

Cases with no expected_sources (pure escalation, ambiguity) are excluded from
recall and reported separately.

KB-ESC-009 is an internal governance document, excluded from the customer
answer pool and reachable only when triage requests it. To measure retrieval
honestly rather than penalising it for a deliberate architectural choice, this
runner simulates that triage step: categories that would trip escalation pass
force_docs. That mirrors production instead of testing a path that does not
exist.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kestrel.embeddings import get_embedder
from kestrel.rerank import get_reranker
from kestrel.retrieval import HybridRetriever
from kestrel.store import get_store

BAR_WIDTH = 24

# Categories where triage would flag escalation and request governance rules.
ESCALATION_CATEGORIES = {"escalate_mandatory", "refuse_scope", "injection"}
GOVERNANCE_DOC = "KB-ESC-009"


def load_cases(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bar(value: float) -> str:
    filled = int(round(value * BAR_WIDTH))
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="evals/eval_set.jsonl")
    ap.add_argument("--db", default="kestrel.db")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "lexical"])
    ap.add_argument("--reranker", default=None,
                    help="noop (default) | cross-encoder")
    ap.add_argument("--min-recall", type=float, default=0.0,
                    help="exit non-zero below this. Use in CI.")
    args = ap.parse_args()

    # Checked before get_store: SQLiteStore's constructor calls sqlite3.connect,
    # which creates the file. Connecting to probe for emptiness would leave a
    # stray zero-chunk database behind and report it as empty.
    if not Path(args.db).exists():
        print(f"Database {args.db} does not exist. Run scripts/ingest.py first.")
        sys.exit(1)

    store = get_store("sqlite", path=args.db)
    embedder = get_embedder(args.provider)
    retriever = HybridRetriever(store, embedder, get_reranker(args.reranker))

    if not retriever.records:
        print(f"No chunks in {args.db}. Run scripts/ingest.py first.")
        sys.exit(1)

    cases = load_cases(args.evals)
    scored = [c for c in cases if c.get("expected_sources")]
    skipped = len(cases) - len(scored)

    per_cat: dict[str, list[float]] = defaultdict(list)
    misses = []

    for case in scored:
        query = f"{case['subject']}\n{case['body']}"
        force = (
            {GOVERNANCE_DOC}
            if case["category"] in ESCALATION_CATEGORIES
            and GOVERNANCE_DOC in case["expected_sources"]
            else None
        )
        got = retriever.retrieve_docs(query, k=args.k, force_docs=force, mode=args.mode)
        expected = set(case["expected_sources"])
        hit = len(expected & set(got)) / len(expected)
        per_cat[case["category"]].append(hit)
        if hit < 1.0:
            misses.append((case["id"], case["category"], sorted(expected - set(got)), got[:3]))

    overall = sum(v for vals in per_cat.values() for v in vals) / len(scored)

    print(f"\nRetrieval eval — mode={args.mode} k={args.k} provider={embedder.name} "
          f"reranker={retriever.reranker.name}")
    print(f"chunks={len(retriever.records)} scored={len(scored)} unscored={skipped}\n")

    for cat in sorted(per_cat):
        vals = per_cat[cat]
        mean = sum(vals) / len(vals)
        print(f"  {cat:<20} {bar(mean)} {mean:6.1%}  (n={len(vals)})")

    print(f"\n  {'OVERALL recall@' + str(args.k):<20} {bar(overall)} {overall:6.1%}\n")

    if misses:
        print(f"Misses ({len(misses)}):")
        for cid, cat, missing, got in misses[:12]:
            print(f"  {cid} [{cat}] missing={missing} got={got}")
        if len(misses) > 12:
            print(f"  ... and {len(misses) - 12} more")

    if overall < args.min_recall:
        print(f"\nFAIL: recall {overall:.1%} below threshold {args.min_recall:.1%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
