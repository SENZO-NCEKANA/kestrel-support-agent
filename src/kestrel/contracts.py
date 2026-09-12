"""
The boundary between model output and the graph.

Everything a model returns arrives here first. The graph downstream may assume
it is dealing with a route from `ROUTES` and a verdict from `VERDICTS`; nothing
upstream may assume the model produced one.

This module exists because `StubLLM` cannot be wrong. It emits fixed strings, so
every parse path in `agent.py` was written against output that was always valid
JSON carrying an always-valid route — and each one defaulted toward `answer` or
`pass` when that assumption broke. A real model breaks it: it returns
`"Escalate"`, or `escalate_mandatory` in the route field, or prose where JSON was
asked for, or nothing at all because the API returned 429.

The rule here is that every degradation resolves toward a human.

    unparseable triage  -> escalate, not answer
    unparseable verdict -> block, not pass
    failed API call     -> escalate / block, not "carry on"

That direction is the whole point. A support agent whose model is down should
stop, not start answering compliance questions from a keyword fallback. Getting
this backwards is not a graceful degradation, it is an outage that answers.

`revise` is treated as blocking. The verifier prompt documents it as "supportable
with the unsupported claims removed", but nothing removes them — a revise loop
that feeds `unsupported_claims` back to the answer node needs a model to be worth
building. Until it exists the choice is between sending a draft the verifier
just said was wrong and escalating, and only one of those is defensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import tools as toolkit

ROUTES = {"answer", "escalate", "refuse", "clarify"}

CATEGORIES = {
    "kb_direct", "kb_multihop", "tool_required", "escalate_mandatory",
    "injection", "trap", "refuse_scope", "ambiguous_clarify",
}

VERDICTS = {"pass", "revise", "block"}

# Verdicts that must not reach the customer as a draft. See the module docstring
# for why `revise` is here rather than driving a rewrite.
BLOCKING_VERDICTS = {"revise", "block"}

# Where anything unrecognised lands.
FAILSAFE_ROUTE = "escalate"
FAILSAFE_CATEGORY = "escalate_mandatory"
FAILSAFE_VERDICT = "block"


@dataclass
class TriageDecision:
    category: str
    route: str
    force_docs: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    reason: str = ""
    degraded: bool = False  # the model's output failed the contract


def _clean(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def parse_triage(data: Any, ok: bool = True) -> TriageDecision:
    """Validate a triage payload, failing closed.

    `ok` is the provider's own success flag: a call that never completed is as
    unusable as one that returned nonsense, and both escalate.

    An unrecognised `category` does not escalate on its own. Category is a label
    for the eval and the trace; `route` is what the graph acts on, and only route
    carries safety weight.
    """
    if not ok:
        return TriageDecision(FAILSAFE_CATEGORY, FAILSAFE_ROUTE,
                              reason="llm call failed", degraded=True)

    if not isinstance(data, dict):
        return TriageDecision(FAILSAFE_CATEGORY, FAILSAFE_ROUTE,
                              reason=f"triage returned {type(data).__name__}, not an object",
                              degraded=True)

    route = _clean(data.get("route"))
    if route not in ROUTES:
        return TriageDecision(
            FAILSAFE_CATEGORY, FAILSAFE_ROUTE,
            force_docs=_string_list(data.get("force_docs")),
            reason=f"unrecognised route {data.get('route')!r}",
            degraded=True,
        )

    category = _clean(data.get("category"))
    if category not in CATEGORIES:
        category = "kb_direct"

    # A hallucinated tool name is dropped at the boundary. Passing it through
    # would reach toolkit.call, which returns an error ToolResult that then gets
    # rendered into the answer context as though it were account data.
    tools = [t for t in _string_list(data.get("expected_tools"))
             if t in toolkit.REGISTRY]

    return TriageDecision(
        category=category,
        route=route,
        force_docs=_string_list(data.get("force_docs")),
        tools=list(dict.fromkeys(tools)),
        reason=str(data.get("reason") or ""),
    )


def parse_verdict(data: Any, ok: bool = True) -> dict:
    """Validate a verifier payload, failing closed.

    Returns a dict that always carries a `verdict` key from `VERDICTS`. The
    inversion from the previous default matters: `resp.data or {"verdict": "pass"}`
    meant a verifier that returned prose, or a list, or nothing, waved the draft
    through. The verifier is the last gate before a customer sees the text, so
    the one thing it must not do is fail silent.
    """
    if not ok:
        return {"verdict": FAILSAFE_VERDICT, "unsupported_claims": [],
                "forbidden_content": [], "missing_escalation": False,
                "notes": "llm call failed; blocked rather than passed"}

    if not isinstance(data, dict) or not data:
        got = "nothing" if not data else type(data).__name__
        return {"verdict": FAILSAFE_VERDICT, "unsupported_claims": [],
                "forbidden_content": [], "missing_escalation": False,
                "notes": f"verifier returned {got}; blocked rather than passed"}

    verdict = _clean(data.get("verdict"))
    if verdict not in VERDICTS:
        return dict(data, verdict=FAILSAFE_VERDICT,
                    notes=f"unrecognised verdict {data.get('verdict')!r}; "
                          "blocked rather than passed")

    return dict(data, verdict=verdict)
