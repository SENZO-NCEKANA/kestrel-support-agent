"""
What happens when the model is wrong.

Every other test in this suite runs on StubLLM, which cannot be wrong: it emits
fixed strings from `_t()`, always valid JSON carrying a route from the allowed
set. That makes it useless for the failure this file is about. A real model
returns "Escalate", or prose where JSON was asked for, or a 429 and nothing at
all — and before contracts.py every one of those defaulted the graph toward
`answer` or `pass`.

ScriptedLLM is the missing half: a provider that returns exactly the malformed
payload each test needs. These assertions are the only evidence the fail-closed
behaviour works until there is a key to point at a real model.
"""
import json
import tempfile
from pathlib import Path

import pytest

from kestrel import tools as toolkit
from kestrel.agent import SAFE_RESPONSES, KestrelAgent
from kestrel.contracts import parse_triage, parse_verdict
from kestrel.embeddings import HashEmbedder
from kestrel.ingest import ingest
from kestrel.llm import LLMResponse, StubLLM
from kestrel.retrieval import HybridRetriever
from kestrel.store import SQLiteStore

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "kb"
EVALS = ROOT / "evals/eval_set.jsonl"

# Distinctive so an assertion can tell "the draft was sent" from "a safe
# response was sent" without matching on prose.
DRAFT = "DRAFT-SENTINEL: the fee is R60 per month."

VALID_TRIAGE = {"category": "kb_direct", "route": "answer",
                "force_docs": [], "expected_tools": [], "reason": "test"}
VALID_VERDICT = {"verdict": "pass", "unsupported_claims": [],
                 "forbidden_content": [], "missing_escalation": False}


class ScriptedLLM:
    """Returns what you queue, including what a model should never return.

    Matches the LLM Protocol. Any task left unspecified gets a valid response,
    so each test overrides only the one call it is about.
    """

    name = "scripted"

    def __init__(self, triage=None, answer=None, verify=None):
        self.responses = {
            "triage": triage or LLMResponse(text=json.dumps(VALID_TRIAGE),
                                            data=dict(VALID_TRIAGE)),
            "answer": answer or LLMResponse(text=DRAFT),
            "verify": verify or LLMResponse(text=json.dumps(VALID_VERDICT),
                                            data=dict(VALID_VERDICT)),
        }
        self.calls: list[str] = []

    def complete(self, system: str, user: str, task: str = "") -> LLMResponse:
        self.calls.append(task)
        return self.responses.get(task, LLMResponse(text=""))


@pytest.fixture(scope="module")
def retriever():
    tmp = tempfile.mkdtemp()
    store = SQLiteStore(str(Path(tmp) / "contract.db"))
    embedder = HashEmbedder()
    ingest(KB, store, embedder)
    return HybridRetriever(store, embedder)


def run_with(retriever, thread_id, **scripted):
    """One ticket through a graph backed by a scripted model."""
    agent = KestrelAgent(retriever, ScriptedLLM(**scripted))
    out = agent.run("Monthly fee on Blue",
                    "What is the monthly account fee on the Kestrel Blue account?",
                    thread_id=thread_id, approve=True)
    return out


# ------------------------------------------------------------------ parsers

def test_parse_triage_rejects_an_unrecognised_route():
    """`escalate_mandatory` is a category, not a route. A model conflating the
    two is the likeliest single malformed-output mode, and it must not answer."""
    d = parse_triage({"route": "escalate_mandatory", "category": "kb_direct"})
    assert d.route == "escalate"
    assert d.degraded


def test_parse_triage_normalises_case_and_whitespace():
    assert parse_triage({"route": " Answer ", "category": "KB_Direct"}).route == "answer"


def test_parse_triage_fails_closed_on_a_failed_call():
    assert parse_triage(None, ok=False).route == "escalate"


def test_parse_triage_fails_closed_on_a_non_dict():
    assert parse_triage([1, 2]).route == "escalate"


def test_parse_triage_drops_hallucinated_tools_and_keeps_real_ones():
    d = parse_triage(dict(VALID_TRIAGE,
                          expected_tools=["get_credit_score", "get_account_profile"]))
    assert d.tools == ["get_account_profile"]


def test_parse_verdict_blocks_on_empty_data():
    """The inversion that matters: this used to default to pass."""
    assert parse_verdict({})["verdict"] == "block"


def test_parse_verdict_blocks_on_a_list_without_raising():
    """json.loads on a JSON array returns a list. dict(list, ...) raises
    TypeError, which used to take the whole run down inside n_verify."""
    assert parse_verdict([1, 2])["verdict"] == "block"


def test_parse_verdict_blocks_on_an_unrecognised_verdict():
    assert parse_verdict({"verdict": "approved"})["verdict"] == "block"


def test_parse_verdict_passes_a_valid_one_through():
    out = parse_verdict(dict(VALID_VERDICT, notes="fine"))
    assert out["verdict"] == "pass" and out["notes"] == "fine"


# -------------------------------------------------------------- through the graph

def test_unknown_route_escalates_and_does_not_send_the_draft(retriever):
    out = run_with(retriever, "c-unknown-route",
                   triage=LLMResponse(text="{}", data=dict(VALID_TRIAGE, route="Escalate!")))
    assert out["route"] == "escalate"
    assert DRAFT not in out["reply"]
    assert out["reply"] == SAFE_RESPONSES["escalate"]
    assert any("DEGRADED" in line for line in out["trace"])


def test_missing_route_key_escalates(retriever):
    payload = {k: v for k, v in VALID_TRIAGE.items() if k != "route"}
    out = run_with(retriever, "c-no-route",
                   triage=LLMResponse(text="{}", data=payload))
    assert out["route"] == "escalate"
    assert DRAFT not in out["reply"]


def test_failed_triage_call_escalates_rather_than_answering(retriever):
    """An API outage must stop the agent, not turn it into a keyword fallback
    that answers compliance questions."""
    out = run_with(retriever, "c-triage-down",
                   triage=LLMResponse(ok=False, error="RateLimitError: 429"))
    assert out["route"] == "escalate"
    assert DRAFT not in out["reply"]
    assert any(c["task"] == "triage" and not c["ok"] for c in out["llm_calls"])


def test_unparseable_verifier_response_blocks_the_draft(retriever):
    """The headline fail-open: OpenAILLM swallows a JSONDecodeError into
    data={}, which used to become a pass and send the draft."""
    out = run_with(retriever, "c-verify-prose",
                   verify=LLMResponse(text="Looks fine to me!", data={}))
    assert out["verdict"]["verdict"] == "block"
    assert DRAFT not in out["reply"]


def test_verifier_returning_a_list_does_not_crash_the_graph(retriever):
    out = run_with(retriever, "c-verify-list",
                   verify=LLMResponse(text="[1,2]", data=[1, 2]))
    assert out["verdict"]["verdict"] == "block"
    assert DRAFT not in out["reply"]


def test_revise_verdict_blocks_rather_than_sending_an_unrevised_draft(retriever):
    """`revise` means "supportable once the unsupported claims are removed", and
    nothing removes them. Until a revise loop exists, sending the draft anyway is
    the one option that is not defensible."""
    out = run_with(retriever, "c-verify-revise",
                   verify=LLMResponse(text="{}", data=dict(VALID_VERDICT, verdict="revise")))
    assert DRAFT not in out["reply"]
    assert out["reply"] == SAFE_RESPONSES["escalate"]


def test_hallucinated_tool_name_is_dropped_before_dispatch(retriever):
    """toolkit.call returns an error ToolResult for an unknown name, which would
    then be rendered into the answer context as though it were account data."""
    toolkit.reset_fixtures()
    out = run_with(retriever, "c-fake-tool",
                   triage=LLMResponse(text="{}", data=dict(
                       VALID_TRIAGE, expected_tools=["get_credit_score"])))
    assert out.get("tool_calls") == []
    assert not out.get("tool_results")


def test_a_valid_tool_still_dispatches(retriever):
    """The filter must not be passing by rejecting everything."""
    toolkit.reset_fixtures()
    out = run_with(retriever, "c-real-tool",
                   triage=LLMResponse(text="{}", data=dict(
                       VALID_TRIAGE, expected_tools=["get_account_profile"])))
    assert any("get_account_profile" in r for r in out["tool_results"])


def test_a_clean_run_still_sends_the_draft(retriever):
    """The control. Fail-closed defaults are worthless if they also block the
    happy path — that would be an outage, not a safety property."""
    out = run_with(retriever, "c-happy")
    assert out["route"] == "answer"
    assert out["verdict"]["verdict"] == "pass"
    assert DRAFT in out["reply"]


# ---------------------------------------------------------------- regression

def test_stub_routing_is_unchanged_across_all_48_cases(retriever):
    """The stub emits valid contracts, so the new fail-closed defaults must be
    invisible to it. A route that moved here is a bug in the parser, not a
    finding about the eval set."""
    toolkit.reset_fixtures()
    agent = KestrelAgent(retriever, StubLLM())
    cases = [json.loads(l) for l in EVALS.read_text().splitlines() if l.strip()]
    misses = []
    for case in cases:
        out = agent.run(case["subject"], case["body"],
                        thread_id=f"reg-{case['id']}", approve=True)
        if out.get("route") != case["expected_route"]:
            misses.append((case["id"], case["expected_route"], out.get("route")))
    assert not misses, f"routing changed under the stub: {misses}"
