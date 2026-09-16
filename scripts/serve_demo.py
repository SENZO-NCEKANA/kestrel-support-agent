#!/usr/bin/env python3
"""
A local page that shows the graph working, one ticket at a time.

    python3 scripts/serve_demo.py                  # offline stub: no key, no cost
    python3 scripts/serve_demo.py --db kestrel-openai.db --provider openai --llm openai

Standard library only. `requirements.txt` keeps onnxruntime and psycopg commented
out to hold CI free and dependency-light, and pulling in a web framework for a
demo would contradict that, so this is `http.server` and nothing else.

The point is the trace, not the reply. A finished answer tells you nothing about
how it was reached, and the two things most worth seeing are invisible in one: a
draft the verifier held back, and an irreversible write pausing for a human. Both
are on the page.

Binds 127.0.0.1 on purpose. Nothing here is hardened for a network, and the
fixtures are not real accounts.
"""
import argparse
import html
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import count
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kestrel import tools as toolkit
from kestrel.agent import build_agent

# Shown beside a draft the graph discarded. The customer never saw this text, and
# the page must never let it read as though they did.
NOT_SENT = "draft withheld — not sent to the customer"

SCENARIOS = {
    "1": ("ATM limit wrong on Private",
          "I am a Kestrel Private customer and I can only draw R2 000 at the ATM. "
          "Your website says Private is R10 000. Please fix this.",
          "ACC-1001"),
    "2": ("Replacement card after fraud",
          "Someone used my card fraudulently and it was cancelled. What will the "
          "replacement card cost me? I am on Blue.",
          "ACC-1002"),
    "3": ("Goods never arrived",
          "I paid for a couch on 20 June and it has never been delivered. It is now "
          "45 days later. Am I too late to dispute it?",
          "ACC-1002"),
    "4": ("Why is my account frozen",
          "My account has been restricted for nine days and nobody will tell me why. "
          "Is this because you reported me to the authorities?",
          "ACC-1003"),
    "5": ("Card stolen, block it now",
          "My wallet was stolen this morning. Please block my card immediately.",
          "ACC-1001"),
}

ACCOUNTS = [
    ("ACC-1001", "Private, verification Level 1"),
    ("ACC-1002", "Blue, verification Level 2"),
    ("ACC-1003", "Plus, Level 3, restricted"),
    ("ACC-1004", "Plus, verification Level 2"),
]

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 64px; font: 15px/1.55 -apple-system, BlinkMacSystemFont,
       "Segoe UI", Roboto, sans-serif; max-width: 900px; margin-inline: auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { opacity: .7; font-size: 13px; margin: 0 0 20px; }
.banner { border: 1px solid currentColor; opacity: .75; border-radius: 6px; padding: 8px 12px;
          font-size: 12.5px; margin-bottom: 20px; }
form.card { border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 16px; margin-bottom: 20px; }
label { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
        opacity: .7; margin-bottom: 4px; }
input[type=text], textarea, select { width: 100%; padding: 8px; font: inherit; margin-bottom: 12px;
        border: 1px solid rgba(128,128,128,.45); border-radius: 5px; background: transparent; color: inherit; }
textarea { min-height: 84px; resize: vertical; }
button { font: inherit; padding: 8px 16px; border-radius: 5px; border: 1px solid rgba(128,128,128,.5);
         background: transparent; color: inherit; cursor: pointer; }
button.primary { border-color: currentColor; font-weight: 600; }
.presets { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }
.presets form { margin: 0; }
.presets button { font-size: 13px; padding: 6px 12px; }
.node { border: 1px solid rgba(128,128,128,.3); border-left-width: 3px; border-radius: 6px;
        padding: 12px 14px; margin-bottom: 10px; }
.node h3 { margin: 0 0 6px; font-size: 13px; text-transform: uppercase; letter-spacing: .05em; opacity: .75; }
.node p { margin: 4px 0; }
.k { opacity: .6; }
pre { white-space: pre-wrap; word-break: break-word; margin: 6px 0 0; font: 13px/1.5 ui-monospace,
      SFMono-Regular, Menlo, monospace; }
.reply { border: 2px solid currentColor; border-radius: 8px; padding: 14px; margin: 18px 0; }
.withheld { border: 1px dashed rgba(128,128,128,.6); border-radius: 8px; padding: 14px; margin: 18px 0; opacity: .85; }
.gate { border: 2px solid currentColor; border-radius: 8px; padding: 16px; margin: 18px 0; }
.gate .row { display: flex; gap: 10px; margin-top: 12px; }
.gate form { margin: 0; }
.tag { display: inline-block; font-size: 12px; padding: 1px 7px; border: 1px solid rgba(128,128,128,.5);
       border-radius: 20px; margin-right: 6px; }
"""


def esc(value) -> str:
    """Everything reaching the page goes through here.

    Ticket bodies are untrusted by design — that is the whole premise of the
    injection work — so rendering one unescaped would be the same mistake in a
    different layer.
    """
    return html.escape("" if value is None else str(value))


def page(body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Kestrel support agent — trace</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>Kestrel support agent</h1>"
        "<p class='sub'>triage → retrieve → [tools] → answer → verify → finalise, "
        "with one rewrite on a revise verdict</p>"
        "<div class='banner'>Kestrel Bank is fictional. The policy corpus is synthetic and the "
        "accounts below are fixtures — no real customer data is involved, and no real card is "
        "ever blocked.</div>"
        f"{body}</body></html>"
    )


def render_form(subject: str = "", body: str = "", account: str = "") -> str:
    options = "".join(
        f"<option value='{esc(acc)}'{' selected' if acc == account else ''}>"
        f"{esc(acc)} — {esc(label)}</option>"
        for acc, label in ACCOUNTS
    )
    presets = "".join(
        "<form method='post' action='/run'>"
        f"<input type='hidden' name='subject' value='{esc(s)}'>"
        f"<input type='hidden' name='body' value='{esc(b)}'>"
        f"<input type='hidden' name='account' value='{esc(a)}'>"
        f"<button type='submit'>Scenario {esc(n)}</button></form>"
        for n, (s, b, a) in SCENARIOS.items()
    )
    return (
        "<form class='card' method='post' action='/run'>"
        "<label for='subject'>Subject</label>"
        f"<input type='text' id='subject' name='subject' value='{esc(subject)}' "
        "placeholder='Card stolen'>"
        "<label for='body'>Body</label>"
        f"<textarea id='body' name='body' placeholder='My wallet was stolen this morning. "
        f"Please block my card immediately.'>{esc(body)}</textarea>"
        "<label for='account'>Account the tools answer as</label>"
        f"<select id='account' name='account'>{options}</select>"
        "<button class='primary' type='submit'>Run the ticket</button></form>"
        f"<div class='presets'>{presets}</div>"
    )


def render_ticket(subject: str, body: str, account: str) -> str:
    return (
        "<div class='node'><h3>Ticket</h3>"
        f"<p><span class='k'>subject</span> {esc(subject)}</p>"
        f"<p><span class='k'>account</span> {esc(account or toolkit.DEFAULT_ACCOUNT)}</p>"
        f"<pre>{esc(body)}</pre></div>"
    )


def render_nodes(state: dict) -> str:
    """The graph's own record of what happened, node by node."""
    triage_route = state.get("triage_route") or state.get("route")
    flag = state.get("injection_flag")
    labels = ", ".join(state.get("injection_labels") or [])
    parts = [
        "<div class='node'><h3>Triage</h3>"
        f"<p><span class='tag'>route {esc(triage_route)}</span>"
        f"<span class='tag'>category {esc(state.get('category'))}</span>"
        + (f"<span class='tag'>injection flagged{': ' + esc(labels) if labels else ''}</span>"
           if flag else "")
        + "</p></div>"
    ]

    citations = state.get("citations") or []
    if citations:
        parts.append(
            "<div class='node'><h3>Retrieve</h3>"
            f"<p>{esc(state.get('n_hits'))} chunks from "
            + "".join(f"<span class='tag'>{esc(c)}</span>" for c in citations)
            + "</p></div>"
        )

    tool_results = state.get("tool_results") or []
    if tool_results:
        parts.append(
            "<div class='node'><h3>Tools</h3>"
            + "".join(f"<pre>{esc(r)}</pre>" for r in tool_results)
            + "</div>"
        )

    verdict = state.get("verdict") or {}
    if verdict:
        claims = verdict.get("unsupported_claims") or []
        revisions = state.get("revisions") or 0
        parts.append(
            "<div class='node'><h3>Verify</h3>"
            f"<p><span class='tag'>verdict {esc(verdict.get('verdict'))}</span>"
            + (f"<span class='tag'>rewritten once</span>" if revisions else "")
            + "</p>"
            + (f"<p>{esc(verdict.get('notes'))}</p>" if verdict.get("notes") else "")
            + "".join(f"<p><span class='k'>unsupported</span> {esc(c)}</p>" for c in claims)
            + "</div>"
        )

    trace = state.get("trace") or []
    if trace:
        parts.append(
            "<div class='node'><h3>Trace</h3><pre>"
            + esc("\n".join(trace))
            + "</pre></div>"
        )
    return "".join(parts)


def render_reply(state: dict) -> str:
    """What the customer actually receives. Never the withheld draft."""
    return (
        "<section class='reply'><h3>Reply sent to the customer</h3>"
        f"<pre>{esc(state.get('reply'))}</pre></section>"
    )


def render_withheld(state: dict) -> str:
    """The draft the graph discarded, when it discarded one.

    Shown because a blocked answer is otherwise a mystery — and marked, so it can
    never be mistaken for what was sent.
    """
    draft = state.get("draft") or ""
    reply = state.get("reply") or ""
    if not draft or reply.startswith(draft):
        return ""
    return (
        f"<section class='withheld'><h3>{esc(NOT_SENT)}</h3>"
        f"<pre>{esc(draft)}</pre></section>"
    )


def render_gate(thread_id: str, state: dict) -> str:
    """The approval interrupt: an irreversible write waiting on a human."""
    requests = []
    for item in state.get("__interrupt__") or []:
        value = getattr(item, "value", item)
        requests.append(value if isinstance(value, dict) else {"action": str(value)})

    rows = "".join(
        f"<p><span class='tag'>{esc(r.get('action'))}</span>"
        f"<span class='k'>account</span> {esc(r.get('account_id'))}</p>"
        f"<p>{esc(r.get('reason'))}</p>"
        for r in requests
    )
    return (
        "<section class='gate'><h3>Waiting for human approval</h3>"
        f"{rows}"
        "<div class='row'>"
        "<form method='post' action='/resume'>"
        f"<input type='hidden' name='thread' value='{esc(thread_id)}'>"
        "<input type='hidden' name='decision' value='approve'>"
        "<button class='primary' type='submit'>Approve</button></form>"
        "<form method='post' action='/resume'>"
        f"<input type='hidden' name='thread' value='{esc(thread_id)}'>"
        "<input type='hidden' name='decision' value='decline'>"
        "<button type='submit'>Decline</button></form>"
        "</div></section>"
    )


def render_result(thread_id: str, subject: str, body: str, account: str, state: dict) -> str:
    """The whole page body for a ticket that has been run."""
    parts = [render_ticket(subject, body, account), render_nodes(state)]
    if state.get("__interrupt__"):
        parts.append(render_gate(thread_id, state))
    else:
        parts.append(render_reply(state))
        parts.append(render_withheld(state))
    parts.append(render_form(subject, body, account))
    return "".join(parts)


def get_route(path: str) -> tuple[str, str]:
    """Where a browser GET should land. Pure, so the routing is testable.

    `/run` and `/resume` take POSTs from the form. A browser arriving at one by
    refreshing after a submit, or by pasting the URL, used to meet a dead-end
    404 — a poor thing to hand someone clicking around a demo. Everything
    unrecognised now goes back to the form instead.
    """
    path = path.split("?", 1)[0]
    if path in ("/", "/index.html"):
        return ("form", "")
    if path.startswith("/t/"):
        return ("result", path[len("/t/"):])
    if path == "/favicon.ico":
        return ("empty", "")
    return ("redirect", "/")


class DemoHandler(BaseHTTPRequestHandler):
    agent = None          # set in main()
    threads: dict = {}    # thread_id -> the ticket it was started from
    results: dict = {}    # thread_id -> (subject, body, account, state)
    counter = count(1)

    def _send(self, body: str, status: int = 200) -> None:
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str) -> None:
        """Post-redirect-get: the result lives at its own URL, so a refresh
        re-renders it instead of re-submitting the ticket."""
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _form(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", "replace")
        return {k: v[0] for k, v in parse_qs(raw, keep_blank_values=True).items()}

    def do_GET(self):  # noqa: N802 — http.server's interface
        kind, arg = get_route(self.path)
        if kind == "form":
            self._send(page(render_form()))
        elif kind == "result":
            stored = self.results.get(arg)
            if not stored:
                self._redirect("/")
                return
            subject, body, account, state = stored
            self._send(page(render_result(arg, subject, body, account, state)))
        elif kind == "empty":
            self.send_response(204)
            self.end_headers()
        else:
            self._redirect(arg)

    def do_POST(self):  # noqa: N802
        if self.path == "/run":
            self._run()
        elif self.path == "/resume":
            self._resume()
        else:
            self._send(page("<p>Not found.</p>"), 404)

    def _run(self) -> None:
        form = self._form()
        subject, body = form.get("subject", "").strip(), form.get("body", "").strip()
        account = form.get("account", "") or toolkit.DEFAULT_ACCOUNT
        if not subject and not body:
            self._redirect("/")
            return

        thread_id = f"ui-{next(self.counter)}"
        self.threads[thread_id] = (subject, body, account)
        # approve=None so a write stops at the gate instead of running: the
        # decision belongs to whoever is looking at the page.
        state = self.agent.run(subject, body, account_id=account,
                               thread_id=thread_id, approve=None)
        self.results[thread_id] = (subject, body, account, state)
        self._redirect(f"/t/{thread_id}")

    def _resume(self) -> None:
        form = self._form()
        thread_id = form.get("thread", "")
        approved = form.get("decision") == "approve"
        if thread_id not in self.threads:
            self._redirect("/")
            return

        subject, body, account = self.threads[thread_id]
        state = self.agent.resume(thread_id, approved)
        self.results[thread_id] = (subject, body, account, state)
        self._redirect(f"/t/{thread_id}")

    def log_message(self, fmt, *args):
        """One tidy line per request instead of http.server's default noise."""
        sys.stderr.write(f"  {self.command} {self.path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default="kestrel.db")
    ap.add_argument("--provider", default=None, help="embedder: hash | openai")
    ap.add_argument("--llm", default=None, help="LLM: stub | openai")
    ap.add_argument("--verifier-model", default=None,
                    help="run the verifier on this model (or LLM_VERIFIER_MODEL)")
    ap.add_argument("--reranker", default=None, help="noop | cross-encoder")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"Database {args.db} does not exist. Run scripts/ingest.py first.")
        sys.exit(1)

    toolkit.reset_fixtures()
    DemoHandler.agent = build_agent(db=args.db, provider=args.provider,
                                    llm_provider=args.llm, k=args.k,
                                    reranker=args.reranker,
                                    verifier_model=args.verifier_model)
    agent = DemoHandler.agent
    model = getattr(agent.llm, "model", "")
    verifier = getattr(agent.verifier_llm, "model", "")
    models = (f" ({model}" + (f", verifier {verifier}" if verifier != model else "") + ")"
              if model else "")

    print(f"\n  Kestrel demo on http://{args.host}:{args.port}")
    print(f"  llm {agent.llm.name}{models}   embedder {agent.retriever.embedder.name}"
          f"   reranker {agent.retriever.reranker.name}")
    print("  Ctrl-C to stop\n")
    try:
        HTTPServer((args.host, args.port), DemoHandler).serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
