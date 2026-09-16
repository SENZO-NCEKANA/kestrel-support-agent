"""
The demo page, tested without a socket.

`scripts/serve_demo.py` keeps its rendering in pure functions that take the state
dict `agent.run` already returns, so everything worth asserting can be checked
offline and for free: that a withheld draft is never presented as the reply, that
an interrupt offers both decisions, and that ticket text is escaped.

That last one is not decoration. Ticket bodies are untrusted input — the premise
of every injection case in the eval set — and rendering one unescaped would be
the same mistake one layer down.
"""
import importlib.util
import tempfile
from pathlib import Path

import pytest

from kestrel import tools as toolkit
from kestrel.agent import KestrelAgent
from kestrel.embeddings import HashEmbedder
from kestrel.ingest import ingest
from kestrel.llm import StubLLM
from kestrel.retrieval import HybridRetriever
from kestrel.store import SQLiteStore

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "kb"

_spec = importlib.util.spec_from_file_location("serve_demo", ROOT / "scripts/serve_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def agent():
    tmp = tempfile.mkdtemp()
    store = SQLiteStore(str(Path(tmp) / "demo.db"))
    embedder = HashEmbedder()
    ingest(KB, store, embedder)
    return KestrelAgent(HybridRetriever(store, embedder), StubLLM())


# ------------------------------------------------------------------ rendering

def test_an_answered_ticket_shows_its_reply_and_what_it_cited(agent):
    out = agent.run("Monthly fee on Plus",
                    "What is the monthly account fee on the Kestrel Plus account?",
                    thread_id="ui-answered")
    html = demo.render_result("ui-answered", "Monthly fee on Plus", "...", "ACC-1001", out)
    assert "Reply sent to the customer" in html
    assert "KB-FEE-001" in html
    assert demo.render_withheld(out) == "", "nothing was withheld, so nothing to show"


def test_a_withheld_draft_is_marked_and_is_never_the_reply(agent):
    """The tipping-off ticket. A draft exists and is discarded; the page has to
    show it — a blocked answer is otherwise a mystery — without ever letting it
    read as what the customer received."""
    out = agent.run("Have you reported me",
                    "Has Kestrel filed a suspicious transaction report about my account?",
                    thread_id="ui-withheld")
    assert out["route"] == "escalate"

    withheld, reply = demo.render_withheld(out), demo.render_reply(out)
    assert demo.NOT_SENT in withheld
    assert out["draft"][:40] in withheld
    assert out["draft"][:40] not in reply


def test_an_interrupt_offers_both_decisions(agent):
    """`approve=None` leaves the write pending, which is the state the page turns
    into two buttons."""
    toolkit.reset_fixtures()
    out = agent.run("Card stolen", "My wallet was stolen. Please block my card.",
                    thread_id="ui-gate", approve=None)
    assert "__interrupt__" in out

    gate = demo.render_gate("ui-gate", out)
    assert "Approve" in gate and "Decline" in gate
    assert "block_card" in gate
    assert "human approval" in gate.lower()
    assert toolkit.CARDS["CRD-5501"]["status"] == "active", "nothing ran while it waited"


def test_ticket_text_is_escaped():
    """A ticket body is untrusted data in the browser exactly as it is in the
    prompt."""
    state = {"subject": "<img src=x onerror=alert(1)>", "body": "<script>alert(1)</script>",
             "route": "answer", "reply": "ok", "trace": []}
    html = demo.render_result("ui-esc", state["subject"], state["body"], "ACC-1001", state)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "onerror=alert(1)>" not in html


def test_rendering_survives_a_partial_state():
    """Every field is read with .get, so a half-finished state renders rather than
    turning a demo into a stack trace in front of whoever is watching."""
    assert demo.render_nodes({}) is not None
    assert "Reply sent to the customer" in demo.render_reply({})
    assert demo.render_withheld({}) == ""
    assert demo.render_result("t", "", "", "", {}) is not None


def test_the_form_names_every_fixture_account():
    """The account decides whether a limit question is capped by tier or by
    verification level, so the page has to make the choice visible."""
    form = demo.render_form()
    for account, _ in demo.ACCOUNTS:
        assert account in form
        assert account in toolkit.ACCOUNTS, f"{account} is not a real fixture"
    assert len(demo.ACCOUNTS) == len(toolkit.ACCOUNTS), "the page lists every fixture, or it misleads"
