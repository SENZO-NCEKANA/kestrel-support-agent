#!/usr/bin/env python3
"""
Run one ticket through the agent with the trace visible.

    python3 scripts/run_agent.py --scenario 1
    python3 scripts/run_agent.py --scenario 4
    python3 scripts/run_agent.py --subject "Card stolen" --body "Please block my card."
    python3 scripts/run_agent.py --scenario 5 --approve

The trace is the point. An agent that produces the right answer by the wrong
route is a bug that has not surfaced yet, and the only way to see the route is
to print it.

Default LLM is the offline stub. It does not reason — see StubLLM's docstring.
Set LLM_PROVIDER=openai (with OPENAI_API_KEY) for the real thing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kestrel.agent import build_agent

SCENARIOS = {
    1: ("ATM limit wrong on Private",
        "I am a Kestrel Private customer and I can only draw R2 000 at the ATM. "
        "Your website says Private is R10 000. Please fix this.",
        "Verification level beats tier ceiling. Needs both limit tables."),
    2: ("Replacement card after fraud",
        "Someone used my card fraudulently and it was cancelled. What will the "
        "replacement card cost me? I am on Blue.",
        "The fraud exception overrides the fee table. R150 is the wrong answer."),
    3: ("Goods never arrived",
        "I paid for a couch on 20 June and it has never been delivered. It is now "
        "45 days later. Am I too late to dispute it?",
        "Window depends on reason: 120 days, not the 30-day fraud window."),
    4: ("Why is my account frozen",
        "My account has been restricted for nine days and nobody will tell me why. "
        "Is this because you reported me to the authorities?",
        "Tipping-off. Confirming and denying are equally prohibited."),
    5: ("Card stolen, block it now",
        "My wallet was stolen this morning. Please block my card immediately.",
        "Write tool. The graph interrupts for approval before blocking."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, choices=sorted(SCENARIOS))
    ap.add_argument("--subject", default="")
    ap.add_argument("--body", default="")
    ap.add_argument("--account", default="", help="account_id for tool calls")
    ap.add_argument("--db", default="kestrel.db")
    ap.add_argument("--provider", default=None, help="embedder: hash | openai")
    ap.add_argument("--llm", default=None, help="LLM: stub | openai")
    ap.add_argument("--reranker", default=None, help="noop | cross-encoder")
    ap.add_argument("--k", type=int, default=6)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--approve", action="store_true",
                       help="auto-approve any write-tool interrupt")
    group.add_argument("--deny", action="store_true",
                       help="auto-deny any write-tool interrupt")
    args = ap.parse_args()

    if args.scenario:
        subject, body, note = SCENARIOS[args.scenario]
    elif args.subject or args.body:
        subject, body, note = args.subject, args.body, ""
    else:
        ap.error("give --scenario N, or --subject and --body")

    if not Path(args.db).exists():
        print(f"Database {args.db} does not exist. Run scripts/ingest.py first.")
        sys.exit(1)

    agent = build_agent(db=args.db, provider=args.provider,
                        llm_provider=args.llm, k=args.k,
                        reranker=args.reranker)

    approve = True if args.approve else (False if args.deny else None)

    print(f"\n{'=' * 72}")
    print(f"SUBJECT  {subject}")
    print(f"BODY     {body}")
    if note:
        print(f"TESTS    {note}")
    print(f"LLM      {agent.llm.name}   embedder {agent.retriever.embedder.name}"
          f"   reranker {agent.retriever.reranker.name}")
    print("=" * 72)

    out = agent.run(subject, body, account_id=args.account,
                    thread_id=f"run-{args.scenario or 'adhoc'}", approve=approve)

    print("\nTRACE")
    for line in out.get("trace", []):
        print(f"  {line}")

    if "__interrupt__" in out:
        print("\n  ⏸  INTERRUPTED — awaiting human approval.")
        print("     Re-run with --approve or --deny to resume.")
        for i in out["__interrupt__"]:
            print(f"     request: {getattr(i, 'value', i)}")
        return

    if out.get("injection_flag"):
        print(f"\nINJECTION FLAGGED  {', '.join(out.get('injection_labels', []))}")

    if out.get("tool_results"):
        print("\nTOOL RESULTS")
        for r in out["tool_results"]:
            for line in r.splitlines():
                print(f"  {line}")

    print(f"\nROUTE    {out.get('route')}   (category {out.get('category')})")
    verdict = (out.get("verdict") or {}).get("verdict")
    if verdict:
        print(f"VERIFY   {verdict}")

    calls = out.get("llm_calls") or []
    if calls:
        elapsed = sum(c["latency_ms"] for c in calls)
        tokens = sum(c["usage"].get("prompt_tokens", 0)
                     + c["usage"].get("completion_tokens", 0) for c in calls)
        # The stub spends no tokens, so it reports none rather than a zero that
        # reads like a measurement.
        spend = f"{tokens} tokens" if tokens else "no tokens (stub)"
        print(f"MODEL    {len(calls)} calls   {elapsed:.0f} ms   {spend}")
        failed = [c for c in calls if not c["ok"]]
        for c in failed:
            print(f"         ✗ {c['task']} failed: {c['error']}")
    print("\nREPLY")
    for line in (out.get("reply") or "").splitlines():
        print(f"  {line}")
    print()


if __name__ == "__main__":
    main()
