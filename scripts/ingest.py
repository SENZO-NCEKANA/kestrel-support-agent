#!/usr/bin/env python3
"""Ingest the knowledge base. Run twice to see incremental skipping work."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kestrel.ingest import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--db", default="kestrel.db")
    ap.add_argument("--provider", default=None, help="hash | openai")
    args = ap.parse_args()

    report, store, _ = build(args.kb, args.db, args.provider)
    print(report)
    if hasattr(store, "count"):
        print(f"store now holds {store.count()} chunks")


if __name__ == "__main__":
    main()
