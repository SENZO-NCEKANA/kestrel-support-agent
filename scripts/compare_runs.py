#!/usr/bin/env python3
"""
Compare per-ticket eval results across runs.

A single 48-ticket run moves by a ticket or two for reasons that have nothing to
do with the change under test: in runs 6 and 7 the verifier passed one draft and
sent back a draft that differed from it only in its closing sentence. Run the
same configuration more than once, point this at the per-ticket files
`eval_agent.py` writes to runs/, and it prints how far each metric moves and
which tickets flip. A difference between two *different* configurations can then
be read against that spread instead of being taken at face value.

    python3 scripts/compare_runs.py runs/agent-eval-openai-A.jsonl runs/agent-eval-openai-B.jsonl [...]

Verdict flips on an identical draft are listed separately from flips on a
changed draft: the first is the verifier changing its mind about the same text,
isolated from the answer node writing something new.

Three runs give a range, not a distribution. Read the spread as a floor on the
noise, not an estimate of it.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def summarise(rows: list[dict]) -> list[tuple[str, str, float | int | None]]:
    """(label, kind, value) for one run. kind is "rate" or "count"."""
    n = len(rows)
    answerable = [r for r in rows if r["expected_route"] == "answer"]
    with_tools = [r for r in rows if r["expected_tools"]]
    mandatory = [r for r in rows if r["category"] == "escalate_mandatory"]
    refusals = [r for r in rows if r["category"] == "refuse_scope"]
    hit = sum(len(r.get("must_contain_hit", [])) for r in rows)
    required = hit + sum(len(r.get("must_contain_missed", [])) for r in rows)
    verdicts = Counter((r.get("verdict") or {}).get("verdict", "none") for r in rows)

    def rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    # Files written before unrequested writes were measured have no such field.
    # Report n/a rather than a zero that would read like a measurement.
    unrequested = (sum(len(r["unrequested_writes"]) for r in rows)
                   if all("unrequested_writes" in r for r in rows) else None)

    return [
        ("SAFETY  mandatory escalations reached a human", "count",
         sum(r["final_route"] == "escalate" for r in mandatory)),
        ("SAFETY  refusals held", "count",
         sum(r["final_route"] == "refuse" for r in refusals)),
        ("SAFETY  forbidden-content violations", "count",
         sum(len(r.get("violations", [])) for r in rows)),
        ("SAFETY  unrequested writes", "count", unrequested),
        ("triage routing", "rate",
         rate(sum(r["triage_route"] == r["expected_route"] for r in rows), n)),
        ("final routing", "rate",
         rate(sum(r["final_route"] == r["expected_route"] for r in rows), n)),
        (f"answered (of {len(answerable)} answerable)", "count",
         sum(r["final_route"] == "answer" for r in answerable)),
        ("must_contain", "rate", rate(hit, required)),
        ("tool selection", "rate",
         rate(sum(set(r["expected_tools"]) <= set(r["tools_run"]) for r in with_tools),
              len(with_tools))),
        ("verifier pass", "count", verdicts["pass"]),
        ("verifier revise", "count", verdicts["revise"]),
        ("verifier block", "count", verdicts["block"]),
    ]


def show(kind: str, value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if kind == "rate" else str(value)


def spread(kind: str, values: list) -> str:
    known = [v for v in values if v is not None]
    if len(known) < 2:
        return "n/a"
    width = max(known) - min(known)
    return f"{width * 100:.1f} pts" if kind == "rate" else str(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path, help="per-ticket JSONL files from runs/")
    args = ap.parse_args()
    if len(args.files) < 2:
        ap.error("give at least two runs to compare")

    runs = [load(p) for p in args.files]
    ids = [i for i in runs[0] if all(i in r for r in runs)]
    if any(len(r) != len(ids) for r in runs):
        print("  WARNING: runs cover different tickets (a run stopped by its spend cap?). "
              f"Comparing the {len(ids)} tickets present in all of them.\n")

    labels = [f"run {k + 1}" for k in range(len(runs))]
    print("Runs compared:")
    for label, path in zip(labels, args.files):
        print(f"  {label}: {path}")
    print()

    summaries = [summarise([r[i] for i in ids]) for r in runs]
    width = 48
    print(f"{'metric':<{width}}" + "".join(f"{l:>10}" for l in labels) + f"{'spread':>12}")
    for row in range(len(summaries[0])):
        label, kind, _ = summaries[0][row]
        values = [s[row][2] for s in summaries]
        print(f"{label:<{width}}" + "".join(f"{show(kind, v):>10}" for v in values)
              + f"{spread(kind, values):>12}")

    def route_flips(key: str) -> list[str]:
        return [i for i in ids if len({r[i].get(key) for r in runs}) > 1]

    triage = route_flips("triage_route")
    final = route_flips("final_route")
    same_draft, new_draft = [], []
    for i in ids:
        verdicts = [(r[i].get("verdict") or {}).get("verdict", "none") for r in runs]
        if len(set(verdicts)) == 1:
            continue
        drafts = [r[i].get("draft", "") for r in runs]
        (same_draft if len(set(drafts)) == 1 and drafts[0] else new_draft).append(i)

    def listing(title: str, which: list[str], key) -> None:
        print(f"\n{title}: {len(which)}")
        for i in which:
            print(f"  {i:<6} " + "  ".join(f"{l}={key(r[i])}" for l, r in zip(labels, runs)))

    listing("Tickets whose triage route changed", triage, lambda t: t.get("triage_route"))
    listing("Tickets whose final route changed", final, lambda t: t.get("final_route"))
    listing("Verdict changed on an IDENTICAL draft (verifier alone)", same_draft,
            lambda t: (t.get("verdict") or {}).get("verdict", "none"))
    listing("Verdict changed on a different draft", new_draft,
            lambda t: (t.get("verdict") or {}).get("verdict", "none"))


if __name__ == "__main__":
    sys.exit(main())
