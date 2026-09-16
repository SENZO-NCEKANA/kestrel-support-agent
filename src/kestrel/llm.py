"""
Pluggable LLM providers, mirroring the embeddings module exactly.

- OpenAILLM — production
- StubLLM   — deterministic, offline, no API key

The stub exists for the same reason HashEmbedder does: so the graph, the routing
and the tool dispatch can be exercised in CI without credentials or spend. It is
a keyword router, not a model.

Read the warning in StubLLM's docstring before quoting any number produced under
it. The stub tells you the plumbing works. It tells you nothing about whether an
agent reasons correctly, and its routing accuracy is a measure of the author's
keyword list, not of anything the system would do in production.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMResponse:
    """Response from the LLM.
    text: The text of the response.
    data: The data of the response.
    tool_calls: The tool calls of the response.
    ok: False when the call never completed. Providers return this instead of
        raising, so one 429 does not kill a 48-case eval run at case 31. The
        graph reads it through kestrel.contracts, which turns a failed call into
        escalate/block rather than letting it default to "answer".
    error: Exception type and message, when ok is False.
    usage: prompt_tokens and completion_tokens. Empty under the stub.
    latency_ms: Wall time for the call.
    """
    text: str = ""
    data: dict = field(default_factory=dict)  # parsed JSON, where the task asks for it
    tool_calls: list[dict] = field(default_factory=list)
    ok: bool = True
    error: str = ""
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, task: str = "") -> LLMResponse: ...


# --------------------------------------------------------------------------- stub

_ESCALATE = [
    (r"suspicious transaction report|reported me|filed a report|report about my account"
     r"|notified the authorities", "fic_report"),
    (r"\bfrozen\b|\brestricted\b|restriction", "restriction"),
    (r"passed away|has died|deceased|my father died|my mother died", "estate"),
    (r"garnishee|court order|emoluments attachment", "court"),
    (r"safe account|scam|tricked me|claiming to be from kestrel", "app_scam"),
    (r"\breporter\b|journalist|national paper|press enquiry", "media"),
    # Widened after EM-07 slipped through: the pattern read "no point in
    # carrying on", the ticket said "do not see the point in carrying on". One
    # word, and the most safety-critical case in the set routed to a fee answer.
    # This list is longer now and still not sufficient — distress does not have
    # a fixed vocabulary, and that is the argument for a model here, not a
    # longer regex. Kept deliberately broad: a false positive costs one
    # unnecessary welfare escalation, a false negative costs far more.
    (r"point in (?:carrying|going) on|end it all|end my life|take my own life"
     r"|kill(?:ing)? myself|harm(?:ing)? myself|hurt myself|suicid"
     r"|not worth living|no reason to (?:live|go on)|cannot go on|can'?t go on"
     r"|nothing left to live for|no way out", "welfare"),
]

_REFUSE = [
    (r"should i (invest|put|move|switch)|what would you do|recommend",
     "fais_advice"),
    (r"tax deductible|declare it to sars|tax treatment", "tax"),
    (r"is that enforceable|enforceable against me|clause \d+", "legal"),
    (r"husband'?s account|wife'?s account|another person'?s account"
     r"|account \d{6}", "third_party"),
]

_TOOLS = [
    (r"my current daily|what is my current|my limit on my account", ["get_account_profile"]),
    (r"do not recognise|dont recognise|unknown charge|charge on my statement",
     ["get_transactions"]),
    (r"\bDSP-\d+|my dispute|lodged a dispute", ["get_dispute_status"]),
    # A write needs a request. This rule used to match "block my card" anywhere,
    # so "How do I block my card?" selected block_card, and the eval — which
    # approves every write — ran it on every build. Tool selection only scores
    # tickets that expect a tool, and the route was right, so nothing saw it
    # until unrequested writes were measured. Request phrasings only.
    (r"please block|card was stolen|wallet was stolen", ["block_card"]),
    (r"am i on the|which account am i", ["get_account_profile"]),
    (r"was .{0,20}charged|did you charge|qualified for the waiver",
     ["get_transactions", "get_account_profile"]),
]


class StubLLM:
    """Deterministic keyword router. CI and tests only.

    WARNING. This is not a model and does not reason. Its triage rules are
    regexes written by the same hand that wrote the eval set, so a routing
    accuracy measured under this stub is close to circular and must never be
    reported as a capability of the system. What it does prove, honestly, is
    that the graph transitions, the tool dispatch, the interrupt and the refusal
    templates all behave — none of which need a model to be worth testing.
    """

    name = "stub"

    def complete(self, system: str, user: str, task: str = "") -> LLMResponse:
        start = time.perf_counter()
        if task == "triage":
            resp = self._triage(user)
        elif task == "answer":
            resp = self._answer(user)
        elif task == "verify":
            resp = self._verify(user)
        else:
            resp = LLMResponse(text="")

        # Timed so the instrumentation path is exercised offline too. This
        # measures graph overhead and nothing else — the stub calls no API and
        # spends no tokens, so `usage` stays empty on purpose rather than
        # reporting a zero that could be mistaken for a measurement.
        resp.latency_ms = (time.perf_counter() - start) * 1000
        return resp

    # -- triage ------------------------------------------------------------
    @staticmethod
    def _unwrap(user: str) -> tuple[str, str]:
        """Recover subject and body from the untrusted-input envelope.

        The envelope is boilerplate. Measuring the ticket's length or scanning
        it for topic words has to happen on the customer's own text, or every
        ticket looks long and specific because the wrapper is.
        """
        subject = body = ""
        for line in user.splitlines():
            if line.startswith("Subject: "):
                subject = line[9:].strip()
            elif line.startswith("Body: "):
                body = line[6:].strip()
        return (subject, body) if (subject or body) else ("", user.strip())

    def _triage(self, user: str) -> LLMResponse:
        subject, body_text = self._unwrap(user)
        low = f"{subject}\n{body_text}".lower()

        for pattern, reason in _ESCALATE:
            if re.search(pattern, low):
                return self._t("escalate_mandatory", "escalate", reason,
                               force=["KB-ESC-009"])

        for pattern, reason in _REFUSE:
            if re.search(pattern, low):
                return self._t("refuse_scope", "refuse", reason,
                               force=["KB-ESC-009"])

        for pattern, tools in _TOOLS:
            if re.search(pattern, low, re.IGNORECASE):
                return self._t("tool_required", "answer", "needs account data",
                               tools=tools)

        # Underspecified: short, and pinned to nothing in particular.
        #
        # The test is for *specifics*, not for topic words. "Please reverse the
        # fee" names a topic and still identifies no fee, no date and no amount,
        # so it cannot be answered without guessing which one. An amount, a
        # date, a reference or a named product is what makes a ticket actionable.
        specific = re.search(
            r"\d|\bblue\b|\bplus\b|\bprivate\b|vault|fixed deposit|swift|atm|"
            r"payshap|debit order|chargeback|ombud|fica|level \w+|"
            r"usd|eur|gbp|aud|jpy|aed", low
        )
        if len(f"{subject} {body_text}".strip()) < 90 and not specific:
            return self._t("ambiguous_clarify", "clarify", "underspecified")

        return self._t("kb_direct", "answer", "answerable from policy")

    @staticmethod
    def _t(category, route, reason, force=None, tools=None) -> LLMResponse:
        data = {
            "category": category,
            "route": route,
            "force_docs": force or [],
            "expected_tools": tools or [],
            "reason": reason,
        }
        return LLMResponse(text=json.dumps(data), data=data)

    # -- answer ------------------------------------------------------------
    def _answer(self, user: str) -> LLMResponse:
        """Echo the retrieved context's provenance. Deliberately not an answer.

        A stub that invented prose would be indistinguishable from a model that
        hallucinated, and would make every groundedness metric meaningless. This
        returns only what it can actually support: which documents were
        retrieved. The real answer node needs a real LLM.
        """
        cites = re.findall(r"--- (\[KB-[^\]]+\][^\n]*) ---", user)
        if not cites:
            return LLMResponse(
                text="I could not find a policy that covers this. Referring it on."
            )
        listed = "\n".join(f"  - {c}" for c in dict.fromkeys(cites))
        return LLMResponse(
            text=("[stub draft — no model called; grounded sources only]\n"
                  f"Relevant policy:\n{listed}")
        )

    # -- verify ------------------------------------------------------------
    def _verify(self, user: str) -> LLMResponse:
        data = {"verdict": "pass", "unsupported_claims": [],
                "forbidden_content": [], "missing_escalation": False,
                "notes": "stub verifier: structural check only, no claim analysis"}
        return LLMResponse(text=json.dumps(data), data=data)


# ------------------------------------------------------------------------ openai

class OpenAILLM:
    name = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # lazy so the package stays optional

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it, or use LLM_PROVIDER=stub "
                "for the offline path."
            )
        # The SDK default timeout is 600s. A support agent that hangs for ten
        # minutes on one ticket has already failed; fail fast and escalate.
        self.client = OpenAI(
            api_key=key,
            timeout=float(os.environ.get("LLM_TIMEOUT", "30")),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "2")),
        )
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str, task: str = "") -> LLMResponse:
        wants_json = task in ("triage", "verify")

        # response_format is omitted rather than set to None. The SDK forwards an
        # explicit None as `"response_format": null`, which the API rejects.
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
        }
        if wants_json:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "") if resp.choices else ""
            usage = {"prompt_tokens": resp.usage.prompt_tokens,
                     "completion_tokens": resp.usage.completion_tokens} if resp.usage else {}
        except Exception as exc:
            # Deliberately broad. Every failure mode here — rate limit, timeout,
            # auth, a malformed response object — has the same correct handling:
            # report it and let contracts.py fail the ticket closed. Raising
            # would abort an eval run partway through and lose the other 47.
            return LLMResponse(ok=False, error=f"{type(exc).__name__}: {exc}",
                               latency_ms=(time.perf_counter() - start) * 1000)

        data = {}
        if wants_json:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            # A JSON array or a bare string parses without error and is still not
            # a triage payload. Passing it through would hand the contract layer
            # a list, where `dict(verdict, ...)` raises TypeError mid-graph.
            data = parsed if isinstance(parsed, dict) else {}

        return LLMResponse(text=text, data=data, usage=usage,
                           latency_ms=(time.perf_counter() - start) * 1000)


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """`model` overrides LLM_MODEL for this one client, so one node can run on a
    different model from the rest."""
    provider = provider or os.environ.get("LLM_PROVIDER", "stub")
    if provider == "openai":
        return OpenAILLM(model)
    if provider == "stub":
        # The stub is not a model. Accepting a model name and ignoring it would
        # label a keyword router's verdicts as a stronger model's.
        if model:
            raise ValueError(f"model {model!r} needs a real provider; the stub is not a model")
        return StubLLM()
    raise ValueError(f"unknown LLM provider: {provider}")
