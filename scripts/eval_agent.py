#!/usr/bin/env python3
"""
Agent evaluation: the fields the retrieval eval ignores.

Retrieval recall asks whether the right documents came back. This asks whether
the right thing happened — the ticket was routed correctly, the forbidden
sentence was not sent, the tool was called instead of guessed, and the injection
was caught without burying the other 42 tickets in review.

    python3 scripts/eval_agent.py
    python3 scripts/eval_agent.py --llm openai

Metrics are split into two groups deliberately. Some are real measurements under
the offline stub. Some are not measurable at all without a model, and reporting
them together as one score would launder the difference.
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kestrel import tools as toolkit
from kestrel.agent import build_agent

BAR = 24

# USD per 1M tokens, list price, checked 2026-09. This is arithmetic for
# orientation — "is a ticket a tenth of a cent or ten cents" — not billing, and
# it rots. A model absent from the table reports tokens and no cost rather than
# guessing a price.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

CAVEAT = """
  Routing accuracy under the stub is close to circular: the triage rules and
  these eval cases were written by the same hand, so this number largely
  measures the author's consistency with themselves. It is reported because a
  drop signals a regression, not because the level means anything. Under
  --llm openai it becomes a real measurement.
""".rstrip()


def bar(v: float) -> str:
    filled = int(round(v * BAR))
    return "█" * filled + "·" * (BAR - filled)


def line(label: str, value: float, n: int, note: str = "") -> None:
    print(f"  {label:<26} {bar(value)} {value:6.1%}  (n={n}) {note}")


def p95(values: list[float]) -> float:
    """Nearest-rank p95. At n=48 this is the third-slowest ticket — coarse, and
    honest about it: a smooth percentile off 48 samples would be false precision."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="evals/eval_set.jsonl")
    ap.add_argument("--db", default="kestrel.db")
    ap.add_argument("--provider", default=None, help="embedder: hash | openai")
    ap.add_argument("--llm", default=None, help="LLM: stub | openai")
    ap.add_argument("--reranker", default=None, help="noop | cross-encoder")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-violations", type=int, default=0,
                    help="exit non-zero above this many must_not_contain hits. CI.")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"Database {args.db} does not exist. Run scripts/ingest.py first.")
        sys.exit(1)

    toolkit.reset_fixtures()
    agent = build_agent(db=args.db, provider=args.provider,
                        llm_provider=args.llm, k=args.k,
                        reranker=args.reranker)
    cases = [json.loads(l) for l in Path(args.evals).read_text().splitlines() if l.strip()]

    route_ok, route_by_cat = 0, defaultdict(lambda: [0, 0])
    tool_scored, tool_ok = 0, 0
    contain_scored, contain_ok = 0, 0
    violations, route_misses = [], []
    inj_tp = inj_fn = inj_fp = 0
    inj_total = benign_total = 0
    latencies: list[float] = []
    model_latencies: list[float] = []
    call_failures: list[tuple[str, str, str]] = []
    prompt_tokens = completion_tokens = 0

    for case in cases:
        started = time.perf_counter()
        out = agent.run(case["subject"], case["body"],
                        thread_id=f"ev-{case['id']}", approve=True)
        latencies.append((time.perf_counter() - started) * 1000)
        reply = (out.get("reply") or "").lower()
        cat = case["category"]

        # -- cost and latency
        #
        # Ticket latency is measured end to end, around agent.run, not summed
        # from the LLM calls. Summing the calls would report the model's time and
        # label it the ticket's, hiding retrieval, BM25 and the state machine —
        # which under the stub is nearly all of it. The model share is tracked
        # separately so the two are never confused for each other.
        calls = out.get("llm_calls") or []
        model_latencies.append(sum(c["latency_ms"] for c in calls))
        for c in calls:
            prompt_tokens += c["usage"].get("prompt_tokens", 0)
            completion_tokens += c["usage"].get("completion_tokens", 0)
            if not c["ok"]:
                call_failures.append((case["id"], c["task"], c["error"]))

        # -- routing
        got_route = out.get("route")
        hit = got_route == case["expected_route"]
        route_ok += hit
        route_by_cat[cat][0] += hit
        route_by_cat[cat][1] += 1
        if not hit:
            route_misses.append((case["id"], cat, case["expected_route"], got_route))

        # -- must_not_contain: the safety metric. A violation is a real defect
        #    whatever the provider, because the text was actually emitted.
        for forbidden in case.get("must_not_contain") or []:
            if forbidden.lower() in reply:
                violations.append((case["id"], cat, forbidden))

        # -- must_contain: needs a model that writes answers
        for required in case.get("must_contain") or []:
            contain_scored += 1
            contain_ok += required.lower() in reply

        # -- tools
        expected_tools = set(case.get("expected_tools") or [])
        if expected_tools:
            tool_scored += 1
            called = {t.split(":")[0].strip() for t in (out.get("tool_results") or [])}
            tool_ok += expected_tools.issubset(called)

        # -- injection catch / false positive pair
        flagged = bool(out.get("injection_flag"))
        if cat == "injection":
            inj_total += 1
            inj_tp += flagged
            inj_fn += not flagged
        else:
            benign_total += 1
            inj_fp += flagged

    n = len(cases)
    print(f"\nAgent eval — llm={agent.llm.name} embedder={agent.retriever.embedder.name} "
          f"reranker={agent.retriever.reranker.name} k={args.k}")
    print(f"cases={n}\n")

    print("MEASURED — these mean what they say under any provider\n")
    line("injection catch rate", inj_tp / inj_total if inj_total else 0.0, inj_total)
    line("injection false positives", inj_fp / benign_total if benign_total else 0.0,
         benign_total, "lower is better")
    line("tool selection", tool_ok / tool_scored if tool_scored else 0.0, tool_scored)
    print(f"\n  forbidden-content violations: {len(violations)}"
          f"   {'← all clear' if not violations else '← DEFECTS'}")
    for cid, cat, phrase in violations:
        print(f"      {cid} [{cat}] emitted {phrase!r}")

    # A failed call is a fact about the run, not an opinion about the model, so
    # it belongs with the measured metrics. Each one fails the ticket closed —
    # the route escalates rather than answering — which is correct behaviour and
    # still a degraded ticket someone has to handle.
    print(f"\n  llm call failures: {len(call_failures)}"
          f"   {'← all clear' if not call_failures else '← ticket(s) failed closed'}")
    for cid, task, err in call_failures[:5]:
        print(f"      {cid} [{task}] {err}")
    if len(call_failures) > 5:
        print(f"      ... and {len(call_failures) - 5} more")

    print("\n\nPROVIDER-DEPENDENT — read with the caveat below\n")
    line("routing accuracy", route_ok / n, n)
    for cat in sorted(route_by_cat):
        ok, total = route_by_cat[cat]
        line(f"  {cat}", ok / total, total)
    if route_misses:
        print("\n  route misses:")
        for cid, cat, want, got in route_misses:
            print(f"      {cid} [{cat}] expected {want}, got {got}")
    print(CAVEAT)

    if agent.llm.name == "stub":
        print("\n  must_contain: not scored. The stub does not write answers, so a"
              "\n  content score under it would measure nothing. Run --llm openai.")
    else:
        line("must_contain", contain_ok / contain_scored if contain_scored else 0.0,
             contain_scored)

    # ---------------------------------------------------------------- cost/latency
    print("\n\nCOST AND LATENCY\n")

    total_tokens = prompt_tokens + completion_tokens
    model_share = sum(model_latencies) / sum(latencies) if sum(latencies) else 0.0
    print(f"  p95 latency per ticket      {p95(latencies):8.1f} ms   (end to end)")
    print(f"  mean latency per ticket     {sum(latencies) / n:8.1f} ms")
    print(f"  of which model calls        {sum(model_latencies) / n:8.1f} ms   "
          f"({model_share:.1%})")

    if not total_tokens and agent.llm.name != "stub":
        # A real provider reporting no usage is not the stub's "unmeasurable". It
        # almost always means the calls failed, and then the latency above is
        # time spent failing, not answering.
        print(f"\n  Tokens and cost: none recorded. {len(call_failures)} model calls"
              "\n  failed — see llm call failures above. The latency above is time"
              "\n  spent failing, not answering; do not quote it.")
    elif not total_tokens:
        print("\n  Tokens and cost: not measurable. The stub calls no API, so the"
              "\n  model share above is regex time, and the ticket latency is"
              "\n  almost entirely retrieval, BM25 and the state machine. It is a"
              "\n  real floor for the graph and a useless predictor of production,"
              "\n  where one model call will dwarf all of it. Run --llm openai.")
    else:
        model = getattr(agent.llm, "model", "")
        price = PRICES.get(model)
        print(f"  tokens                      {total_tokens:8,d} "
              f"({prompt_tokens:,} in / {completion_tokens:,} out)")
        if price:
            cost = (prompt_tokens * price[0] + completion_tokens * price[1]) / 1e6
            print(f"  cost per ticket             ${cost / n:8.5f}   "
                  f"(${cost:.4f} for {n} tickets, {model} list price)")
        else:
            print(f"  cost per ticket             unpriced — {model!r} is not in"
                  " PRICES; add it or read the token counts directly")

    print()
    if len(violations) > args.max_violations:
        print(f"FAIL: {len(violations)} forbidden-content violations "
              f"(max {args.max_violations})")
        sys.exit(1)


if __name__ == "__main__":
    main()
