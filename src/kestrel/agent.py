"""
The support agent as a state machine.

triage -> retrieve -> [tools] -> answer -> verify -> finalise
                                    ^        |
                                    +--------+  one rewrite on `revise`

Safety lives in the shape of the graph, not in the wording of a prompt. A
tipping-off question is routed to escalate by triage and never reaches the
answer node at all, so there is no draft for a model to be talked out of. The
alternative — one node with a long prompt listing what not to say — puts the
rule and the temptation in the same place.

Three properties this structure buys:

1. Triage can terminate. Escalation and refusal are routes, not instructions.
2. The write tool interrupts. block_card suspends the graph for human approval
   before it runs, because it cannot be undone. Once approved and run, the reply
   says so, even when the ticket escalates — a customer is never told no decision
   was made after one was.
3. Verify can veto. A draft that reaches finalise carrying forbidden content is
   discarded and replaced with a safe response, however fluent it was.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import injection, tools as toolkit
from .contracts import (BLOCKING_VERDICTS, FAILSAFE_VERDICT, parse_triage,
                        parse_verdict)
from .llm import LLM, LLMResponse, get_llm
from .retrieval import HybridRetriever, format_context

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"

# Routes that must never produce a drafted answer.
TERMINAL_ROUTES = {"escalate", "refuse", "clarify"}

# How many times a draft may go back to the answer node on a `revise` verdict.
# One. The loop is an extra chance, never a weakening: a ticket that uses it up
# finalises exactly as it did before the loop existed, because `revise` is still
# a blocking verdict at finalise. The count is incremented in n_answer rather
# than on the edge, so the bound holds even if the edge logic changes later, and
# a verifier stuck on `revise` costs one extra pass rather than a run's budget.
MAX_REVISIONS = 1

# The internal escalation and routing matrix. It reaches the answer layer through
# force_docs, and triage and the verifier read sections of it directly.
GOVERNANCE_DOC = "KB-ESC-009"

# The matrix sections triage routes on. Confidence Routing is left out on
# purpose: it depends on retrieval results triage has not seen yet, and
# "conflicting policies retrieved: escalate" would push triage toward escalating
# more, the failure this exists to fix.
#
# Untrusted Input was left out too, on the reasoning that the rule-based filter
# and the untrusted-ticket envelope already enforced it. Seven real-model runs
# disagreed: IJ-02, IJ-03 and IJ-05 escalated in every one of them. The filter
# flags the attempt and the envelope marks the text as data, but neither tells
# triage what route a flagged ticket takes — so triage invented one, and the safe
# invention is always escalation. The section says plainly that an injection is
# not automatically a fraudulent ticket and that the legitimate request is still
# served, which is exactly the decision triage was missing.
TRIAGE_MATRIX_SECTIONS = (
    "Purpose and Precedence",
    "Mandatory Escalation",
    "Mandatory Refusal, With Referral",
    "Topics an Agent May Answer Fully",
    "Untrusted Input",
)

# The verifier no longer judges escalation. In the second real run it blocked
# correct answers as missed escalations; given the matrix in the third, it
# blocked just as often, citing triggers that did not fit the tickets. Escalation
# is triage's decision, and the graph already forces a block on any draft for a
# ticket triage escalated or refused (see n_verify). The verifier is left with
# what only it can judge — groundedness, forbidden content, and whether a draft
# obeyed an injected instruction — and keeps the one section that bears on that.
VERIFIER_MATRIX_SECTIONS = ("Untrusted Input",)

SAFE_RESPONSES = {
    "escalate": (
        "I am not able to discuss this on the ticket. I have referred it to the "
        "specialist team that handles it, and they will contact you directly. "
        "I have not made any decision about your account and I am not able to "
        "give you a reason here."
    ),
    # Used when an approved write has already run. The standard escalation text
    # says no decision was made about the account, which is then false.
    "escalate_after_action": (
        "I am not able to discuss the rest of this on the ticket. I have referred "
        "it to the specialist team that handles it, and they will contact you "
        "directly."
    ),
    "refuse": (
        "This falls outside what I am able to advise on. I have set out below "
        "who can help, and I have not given an opinion either way."
    ),
    "clarify": (
        "I want to get this right rather than guess. Could you tell me a little "
        "more about what happened, and when?"
    ),
}

# What the customer is told about a write that actually ran. Worded to stay true
# when the card was already blocked and the tool changed nothing.
ACTION_NOTICES = {
    "block_card": "Your card is now blocked and can no longer be used.",
}
GENERIC_ACTION_NOTICE = "I have completed the change you asked for on your account."


class TicketState(TypedDict, total=False):
    subject: str
    body: str
    account_id: str
    category: str
    route: str
    # Triage's own decision. `route` is rewritten by n_finalise when the verifier
    # blocks, so scoring it alone would blame triage for the verifier's strictness.
    triage_route: str
    force_docs: list[str]
    injection_flag: bool
    injection_labels: list[str]
    n_hits: int
    context: str
    citations: list[str]
    tool_calls: list[str]
    tool_results: list[str]
    # Customer-facing notices for write tools that actually ran.
    actions_taken: list[str]
    approved: bool | None
    draft: str
    # Rewrites this ticket has had after a `revise` verdict. Bounded by
    # MAX_REVISIONS, and read by the edge out of verify.
    revisions: int
    verdict: dict
    reply: str
    trace: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    llm_calls: Annotated[list[dict], lambda a, b: (a or []) + (b or [])]


def _read(name: str) -> str:
    path = PROMPTS / name
    return path.read_text() if path.exists() else ""


def _chunk_index(chunk_id: str) -> int:
    """Position within its document: `KB-ESC-009#3-1003e9ac` -> 3."""
    try:
        return int(chunk_id.split("#", 1)[1].split("-", 1)[0])
    except (IndexError, ValueError):
        return 0


def _matrix_prompt(records, sections: tuple[str, ...]) -> str:
    """Render the named matrix sections for a node's system prompt.

    Loaded from the store rather than copied into a prompt file, so the matrix
    stays the single source of truth and a governance edit reaches every node
    that reads it on re-ingest.

    The first real-model run escalated fee, dispute and account-data questions
    because triage was told to escalate what "must not be answered" and never
    shown what that meant. So a store that has the matrix but lacks a required
    section refuses to build an agent: running without the rules a node was
    written against is the failure, and it is quieter than an error. A store with
    no matrix at all builds normally.
    """
    chunks = sorted((r for r in records if r.doc_id == GOVERNANCE_DOC),
                    key=lambda r: _chunk_index(r.chunk_id))
    if not chunks:
        return ""

    by_heading: dict[str, list[str]] = {}
    for chunk in chunks:
        by_heading.setdefault(chunk.heading_path, []).append(chunk.text.strip())

    missing = [h for h in sections if h not in by_heading]
    if missing:
        raise ValueError(
            f"{GOVERNANCE_DOC} is in the store but has no section for {missing}. "
            "Triage and the verifier are written against these sections; building "
            "an agent without them reintroduces the failures they exist to prevent."
        )

    rendered = [f"### {h}\n\n" + "\n\n".join(by_heading[h]) for h in sections]
    return "\n\n## Escalation and routing matrix\n\n" + "\n\n".join(rendered) + "\n"


def _account_data(state: TicketState) -> str:
    """The tool results as the answer and verify nodes both see them.

    One function so the two cannot drift apart. A draft built from account data
    has to be judged against that same data: in run 6 the verifier sent back a
    correct answer — the customer's tier and its benefits — because the tool
    result that established the tier never reached it.
    """
    if not state.get("tool_results"):
        return ""
    return "--- account data (from tools) ---\n" + "\n".join(state["tool_results"])


def _objection(state: TicketState) -> str:
    """The verifier's objection, rendered for a rewrite.

    Only what the verifier actually said — its notes and the claims it could not
    find support for. Nothing is paraphrased or added: a rewrite prompt that
    restated the objection in its own words would be a third opinion about the
    draft, and the answer node would be rewriting against the wrong one.
    """
    verdict = state.get("verdict") or {}
    if verdict.get("verdict") != "revise":
        return ""

    lines = []
    if verdict.get("notes"):
        lines.append(str(verdict["notes"]).strip())
    claims = verdict.get("unsupported_claims")
    if isinstance(claims, list):
        lines += [f"- unsupported: {c}" for c in claims if isinstance(c, str) and c.strip()]
    if not lines:
        return ""
    return "--- verifier objection (rewrite the draft) ---\n" + "\n".join(lines)


class KestrelAgent:
    def __init__(self, retriever: HybridRetriever, llm: LLM | None = None,
                 k: int = 6, require_approval: bool = True,
                 verifier_llm: LLM | None = None):
        self.retriever = retriever
        self.llm = llm or get_llm()
        # The verifier can run on a different model from triage and the answer.
        # Twice it was given better inputs and did not improve (runs 3 and 7); the
        # one change that helped took a job away from it. A stronger model on this
        # node alone tests whether the small model is the bias.
        self.verifier_llm = verifier_llm or self.llm
        self.k = k
        self.require_approval = require_approval
        records = retriever.all_records
        self.prompts = {
            "triage": _read("triage.md") + _matrix_prompt(records, TRIAGE_MATRIX_SECTIONS),
            "answer": _read("answer.md"),
            "verify": _read("verifier.md") + _matrix_prompt(records, VERIFIER_MATRIX_SECTIONS),
        }
        self.graph = self._build()

    # ------------------------------------------------------------- nodes
    def n_triage(self, state: TicketState) -> dict:
        subject, body = state.get("subject", ""), state.get("body", "")

        # Runs on raw text, before any model sees it.
        verdict = injection.detect(f"{subject}\n{body}")

        wrapped = injection.wrap_untrusted(subject, body)
        resp = self.llm.complete(self.prompts["triage"], wrapped, task="triage")

        # Nothing below may assume the model returned a usable route. An
        # unrecognised one, or a call that never completed, becomes `escalate` —
        # never `answer`. See contracts.py for why that direction is the only
        # defensible one.
        decision = parse_triage(resp.data, ok=resp.ok)
        force = list(decision.force_docs)

        # An injection attempt is flagged but does not by itself change the
        # route: the legitimate request underneath still deserves service.
        # Governance rules are pulled in so the answer layer sees them.
        if verdict.flagged and GOVERNANCE_DOC not in force:
            force.append(GOVERNANCE_DOC)

        trace = [f"triage: category={decision.category} route={decision.route} "
                 f"injection={verdict.flagged} force={force or '-'}"]
        if decision.degraded:
            trace.append(f"triage: DEGRADED — {decision.reason}; "
                         f"failed closed to {decision.route}")

        return {
            "category": decision.category,
            "route": decision.route,
            "triage_route": decision.route,
            "force_docs": force,
            "injection_flag": verdict.flagged,
            "injection_labels": verdict.labels,
            "tool_calls": decision.tools,
            "llm_calls": [_call_record("triage", resp, self.llm)],
            "trace": trace,
        }

    def n_retrieve(self, state: TicketState) -> dict:
        query = f"{state.get('subject','')}\n{state.get('body','')}"
        force = set(state.get("force_docs") or []) or None
        hits = self.retriever.retrieve(query, k=self.k, force_docs=force)
        cites = list(dict.fromkeys(h.doc_id for h in hits))
        # Only the rendered context and the doc ids go into state. Putting the
        # Hit objects themselves in would make the checkpointer serialise
        # project-internal dataclasses on every step, which langgraph warns
        # about and which nothing downstream reads.
        return {
            "n_hits": len(hits),
            "context": format_context(hits),
            "citations": cites,
            "trace": [f"retrieve: {len(hits)} chunks from {cites}"],
        }

    def n_tools(self, state: TicketState) -> dict:
        account = state.get("account_id") or toolkit.DEFAULT_ACCOUNT
        results, actions, trace = [], [], []

        for name in state.get("tool_calls") or []:
            if name in toolkit.WRITE_TOOLS and self.require_approval:
                decision = interrupt({
                    "action": name,
                    "account_id": account,
                    "reason": "irreversible write — human approval required",
                })
                approved = decision is True or (
                    isinstance(decision, dict) and decision.get("approved") is True
                )
                if not approved:
                    trace.append(f"tools: {name} DENIED by approver")
                    results.append(f"{name}: not executed, approval denied")
                    continue
                trace.append(f"tools: {name} approved")

            kwargs: dict[str, Any] = {"account_id": account}
            if name == "get_dispute_status":
                kwargs = {"reference": _find_dispute_ref(state)}
            result = toolkit.call(name, **kwargs)
            results.append(result.render())
            trace.append(f"tools: {name} ok={result.ok}")

            # A write that ran changed the account. Whatever the route ends up
            # being, the customer has to be told.
            if name in toolkit.WRITE_TOOLS and result.ok:
                actions.append(ACTION_NOTICES.get(name, GENERIC_ACTION_NOTICE))

        return {"tool_results": results, "actions_taken": actions, "trace": trace}

    def n_answer(self, state: TicketState) -> dict:
        parts = [state.get("context", "")]
        account_data = _account_data(state)
        if account_data:
            parts.append(account_data)
        parts.append(injection.wrap_untrusted(state.get("subject", ""),
                                              state.get("body", "")))

        # On a rewrite the verifier's objection goes in beside the same context
        # the faulted draft was written from. It names the claim; the answer node
        # grounds it or drops it. See the Revision section of answer.md.
        #
        # The pass is counted on the verdict, not on whether an objection rendered:
        # a `revise` carrying no notes and no claims would otherwise increment
        # nothing and the graph would circle between answer and verify.
        revising = (state.get("verdict") or {}).get("verdict") == "revise"
        objection = _objection(state) if revising else ""
        if objection:
            parts.append(objection)

        resp = self.llm.complete(self.prompts["answer"], "\n\n".join(parts), task="answer")
        return {"draft": resp.text,
                "revisions": (state.get("revisions") or 0) + (1 if revising else 0),
                "llm_calls": [_call_record("answer", resp, self.llm)],
                "trace": [f"answer: {'redrafted' if revising else 'drafted'} "
                          f"{len(resp.text)} chars"]}

    def n_verify(self, state: TicketState) -> dict:
        # The verifier judges the draft's content: whether its claims are grounded,
        # whether it says something forbidden, and whether it obeyed an instruction
        # in the ticket — which it can only judge against the ticket itself, sent
        # in the same untrusted envelope the other nodes use. Whether the ticket
        # should have been escalated is not its call; the override below enforces
        # that structurally.
        #
        # It is shown the account data the draft was written from, when there is
        # any. A claim about the customer's own account can only be grounded there,
        # never in the policy context and never in the ticket.
        parts = [f"--- context ---\n{state.get('context','')}"]
        account_data = _account_data(state)
        if account_data:
            parts.append(account_data)
        parts.append(f"--- draft ---\n{state.get('draft','')}")
        parts.append("--- ticket ---\n"
                     + injection.wrap_untrusted(state.get("subject", ""),
                                                state.get("body", "")))
        resp = self.verifier_llm.complete(self.prompts["verify"], "\n\n".join(parts),
                                          task="verify")

        # The last gate before a customer sees the text, so the one thing it must
        # not do is fail silent. A verifier that returns prose, a list, or
        # nothing at all blocks the draft rather than waving it through.
        verdict = parse_verdict(resp.data, ok=resp.ok)

        # Structural override the stub cannot reason its way to: if triage said
        # escalate and a draft exists anyway, that is a missed escalation
        # regardless of what the verifier thinks of the prose.
        if state.get("route") in TERMINAL_ROUTES and state.get("draft"):
            verdict = dict(verdict, verdict="block", missing_escalation=True)

        return {"verdict": verdict,
                "llm_calls": [_call_record("verify", resp, self.verifier_llm)],
                "trace": [f"verify: {verdict.get('verdict')}"]}

    def n_finalise(self, state: TicketState) -> dict:
        route = state.get("route", "answer")
        # Defaults point toward a human here as everywhere else: a missing
        # verdict is an absent gate, not a passed one.
        verdict = (state.get("verdict") or {}).get("verdict", FAILSAFE_VERDICT)

        # `revise` still blocks here, and that is not the same as ignoring it. A
        # draft arriving on a revise verdict has already had its one rewrite, or
        # was never eligible for one (see _after_verify), so the claims the
        # verifier faulted are still in it. Sending that is the one option that
        # is not defensible.
        if route in TERMINAL_ROUTES or verdict in BLOCKING_VERDICTS:
            final_route = route if route in TERMINAL_ROUTES else "escalate"
            note = (f"{verdict} by verifier"
                    if verdict in BLOCKING_VERDICTS and route == "answer" else route)

            # An approved write already ran. The draft that would have mentioned
            # it is being discarded, so the safe response carries it instead —
            # and must not use the escalation text that says no decision was made.
            actions = state.get("actions_taken") or []
            if actions:
                body = SAFE_RESPONSES["escalate_after_action" if final_route == "escalate"
                                      else final_route]
                reply = " ".join(actions) + " " + body
                note += f", after {len(actions)} completed action(s)"
            else:
                reply = SAFE_RESPONSES[final_route]

            return {"reply": reply, "route": final_route,
                    "trace": [f"finalise: safe response ({note})"]}

        reply = state.get("draft", "")
        if state.get("citations"):
            reply += "\n\nSources: " + ", ".join(state["citations"])
        return {"reply": reply, "trace": ["finalise: answered"]}

    def n_clarify(self, state: TicketState) -> dict:
        return {"reply": SAFE_RESPONSES["clarify"], "trace": ["clarify: asked for detail"]}

    # ------------------------------------------------------------- edges
    @staticmethod
    def _after_triage(state: TicketState) -> str:
        return "clarify" if state.get("route") == "clarify" else "retrieve"

    @staticmethod
    def _after_retrieve(state: TicketState) -> str:
        return "tools" if state.get("tool_calls") else "answer"

    @staticmethod
    def _after_tools(state: TicketState) -> str:
        return "answer"

    @staticmethod
    def _after_verify(state: TicketState) -> str:
        """One rewrite on a `revise`, then finalise whatever comes back.

        Three conditions, all required. `block` never loops: forbidden content or
        an obeyed injection is not a wording problem, and a failed verifier call
        fails closed to `block`, so a verifier that is down cannot drive rewrites.
        A terminal route never loops: triage escalated or refused the ticket, no
        draft may reach the customer, and n_verify has already blocked it. And the
        bound is hard, so a model that answers `revise` forever costs one extra
        pass rather than a run.
        """
        verdict = (state.get("verdict") or {}).get("verdict", FAILSAFE_VERDICT)
        if (verdict == "revise"
                and (state.get("revisions") or 0) < MAX_REVISIONS
                and state.get("route") not in TERMINAL_ROUTES):
            return "answer"
        return "finalise"

    def _build(self):
        g = StateGraph(TicketState)
        g.add_node("triage", self.n_triage)
        g.add_node("retrieve", self.n_retrieve)
        g.add_node("tools", self.n_tools)
        g.add_node("answer", self.n_answer)
        g.add_node("verify", self.n_verify)
        g.add_node("finalise", self.n_finalise)
        g.add_node("clarify", self.n_clarify)

        g.add_edge(START, "triage")
        g.add_conditional_edges("triage", self._after_triage,
                                {"clarify": "clarify", "retrieve": "retrieve"})
        g.add_conditional_edges("retrieve", self._after_retrieve,
                                {"tools": "tools", "answer": "answer"})
        g.add_edge("tools", "answer")
        g.add_edge("answer", "verify")
        g.add_conditional_edges("verify", self._after_verify,
                                {"answer": "answer", "finalise": "finalise"})
        g.add_edge("finalise", END)
        g.add_edge("clarify", END)

        return g.compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------- public
    def run(self, subject: str, body: str, account_id: str = "",
            thread_id: str = "t1", approve: bool | None = None) -> dict:
        """Run one ticket. Returns the final state.

        If the graph interrupts for approval, `approve` decides: True resumes
        and executes, False resumes and declines. None leaves it interrupted and
        the returned state carries `__interrupt__`.
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}
        state = {"subject": subject, "body": body, "account_id": account_id}
        out = self.graph.invoke(state, config=config)

        if "__interrupt__" in out and approve is not None:
            out = self.graph.invoke(Command(resume=approve), config=config)
        return out

    def resume(self, thread_id: str, approved: bool) -> dict:
        """Resume a thread stopped at the approval interrupt, with a decision.

        `run` always starts a ticket from the beginning, which is right when the
        decision is known up front. It is wrong when the approval arrives later
        and separately — a second `run` would hand a fresh input to a graph that
        is paused mid-step. The demo server needs exactly that: the interrupt is
        one HTTP request and the decision is the next.
        """
        from langgraph.types import Command

        return self.graph.invoke(
            Command(resume=approved),
            config={"configurable": {"thread_id": thread_id}},
        )


def _call_record(task: str, resp: LLMResponse, llm: LLM) -> dict:
    """One row of per-ticket LLM instrumentation.

    A plain dict rather than a dataclass for the same reason `n_retrieve` keeps
    Hit objects out of state: the checkpointer serialises state on every step,
    and project-internal types there are what langgraph warns about.

    `model` names what served the call, because the nodes need not share one and
    the eval prices each call at its own model's rate.
    """
    return {"task": task, "model": getattr(llm, "model", "") or llm.name,
            "ok": resp.ok, "latency_ms": round(resp.latency_ms, 2),
            "usage": dict(resp.usage), "error": resp.error}


def _find_dispute_ref(state: TicketState) -> str:
    import re
    m = re.search(r"DSP-\d+", f"{state.get('subject','')} {state.get('body','')}",
                  re.IGNORECASE)
    return m.group(0).upper() if m else ""


def build_agent(db: str = "kestrel.db", provider: str | None = None,
                llm_provider: str | None = None, k: int = 6,
                require_approval: bool = True,
                reranker: str | None = None,
                verifier_model: str | None = None) -> KestrelAgent:
    import os

    from .embeddings import get_embedder
    from .rerank import get_reranker
    from .store import get_store

    retriever = HybridRetriever(get_store("sqlite", path=db), get_embedder(provider),
                                get_reranker(reranker))
    verifier_model = verifier_model or os.environ.get("LLM_VERIFIER_MODEL")
    verifier_llm = get_llm(llm_provider, model=verifier_model) if verifier_model else None
    return KestrelAgent(retriever, get_llm(llm_provider), k=k,
                        require_approval=require_approval, verifier_llm=verifier_llm)
