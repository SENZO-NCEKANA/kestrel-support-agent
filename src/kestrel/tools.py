"""
Mock banking tools with deterministic fixtures.

These exist because the knowledge base cannot answer account-specific questions.
The limits policy gives a cap per tier *and* per verification level; which one
binds for this customer is not in any document. An agent that guesses it is
inventing data, and `must_not_contain` on the tool_required eval cases is there
to catch exactly that.

Fixtures are fixed data, not a simulation. Same input, same output, forever, so
a test that depends on a balance is not depending on a random number.

One tool writes. `block_card` is irreversible in the real system — a blocked
card is replaced, never unblocked — so it is the tool the graph interrupts on
for human approval. An approval gate guarding only reads would be decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# Deterministic account fixtures. ACC-1001 is the scenario-1 customer: Private
# tier, but verified only to Level 1, so the verification cap binds and the tier
# ceiling the pricing page advertises is unreachable.
ACCOUNTS: dict[str, dict] = {
    "ACC-1001": {
        "account_id": "ACC-1001", "name": "T. Mokoena", "tier": "Private",
        "verification_level": 1, "status": "active",
        "card_id": "CRD-5501", "opened": "2024-11-02",
    },
    "ACC-1002": {
        "account_id": "ACC-1002", "name": "S. Naidoo", "tier": "Blue",
        "verification_level": 2, "status": "active",
        "card_id": "CRD-5502", "opened": "2023-06-14",
    },
    "ACC-1003": {
        "account_id": "ACC-1003", "name": "P. van Wyk", "tier": "Plus",
        "verification_level": 3, "status": "restricted",
        "card_id": "CRD-5503", "opened": "2022-01-30",
    },
    # A Plus account that is not restricted. ACC-1003 is restricted on purpose,
    # for the tickets about a frozen account; a fee or benefits question asked by
    # a Plus customer needs an account where nothing else is going on.
    "ACC-1004": {
        "account_id": "ACC-1004", "name": "L. Dlamini", "tier": "Plus",
        "verification_level": 2, "status": "active",
        "card_id": "CRD-5504", "opened": "2025-03-19",
    },
}

TRANSACTIONS: dict[str, list[dict]] = {
    "ACC-1001": [
        {"date": "2026-08-04", "description": "Unpaid debit order fee", "amount": -85.00,
         "reference": "TXN-88412"},
        {"date": "2026-08-04", "description": "MTN debit order returned", "amount": 0.00,
         "reference": "TXN-88411"},
        {"date": "2026-08-01", "description": "Monthly account fee - Private", "amount": -395.00,
         "reference": "TXN-88101"},
        {"date": "2026-07-28", "description": "ATM withdrawal Sandton", "amount": -2000.00,
         "reference": "TXN-87990"},
    ],
    "ACC-1002": [
        {"date": "2026-08-01", "description": "Monthly account fee - Blue", "amount": -60.00,
         "reference": "TXN-88102"},
        {"date": "2026-06-20", "description": "Loungeware Furniture", "amount": -8499.00,
         "reference": "TXN-85220"},
    ],
    "ACC-1003": [
        {"date": "2026-08-01", "description": "Monthly account fee - Plus", "amount": -135.00,
         "reference": "TXN-88103"},
    ],
    "ACC-1004": [
        {"date": "2026-08-01", "description": "Monthly account fee - Plus", "amount": -135.00,
         "reference": "TXN-88104"},
        {"date": "2026-07-22", "description": "Card purchase - Woolworths", "amount": -1240.50,
         "reference": "TXN-87540"},
    ],
}

DISPUTES: dict[str, dict] = {
    "DSP-40192": {
        "reference": "DSP-40192", "account_id": "ACC-1002",
        "reason": "Goods or services not received", "transaction_reference": "TXN-85220",
        "transaction_date": "2026-06-20", "opened": "2026-08-04",
        "state": "awaiting merchant response", "provisional_credit": False,
    },
    "DSP-40155": {
        "reference": "DSP-40155", "account_id": "ACC-1001",
        "reason": "Unauthorised or fraudulent transaction", "transaction_reference": "TXN-84001",
        "transaction_date": "2026-07-15", "opened": "2026-07-16",
        "state": "provisional credit raised", "provisional_credit": True,
    },
}

CARDS: dict[str, dict] = {
    "CRD-5501": {"card_id": "CRD-5501", "account_id": "ACC-1001", "status": "active"},
    "CRD-5502": {"card_id": "CRD-5502", "account_id": "ACC-1002", "status": "active"},
    "CRD-5503": {"card_id": "CRD-5503", "account_id": "ACC-1003", "status": "active"},
    "CRD-5504": {"card_id": "CRD-5504", "account_id": "ACC-1004", "status": "active"},
}

DEFAULT_ACCOUNT = "ACC-1001"


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: dict = field(default_factory=dict)
    error: str = ""

    def render(self) -> str:
        if not self.ok:
            return f"{self.tool}: ERROR {self.error}"
        rows = "\n".join(f"  {k}: {v}" for k, v in self.data.items())
        return f"{self.tool}:\n{rows}"


# ------------------------------------------------------------------- read tools

def get_account_profile(account_id: str = DEFAULT_ACCOUNT, **_) -> ToolResult:
    """Tier, FICA verification level and status. The KB cannot supply these."""
    acc = ACCOUNTS.get(account_id)
    if not acc:
        return ToolResult("get_account_profile", False, error=f"no account {account_id}")
    return ToolResult("get_account_profile", True, data=dict(acc))


def get_transactions(account_id: str = DEFAULT_ACCOUNT, days: int = 90,
                     search: str = "", **_) -> ToolResult:
    if account_id not in ACCOUNTS:
        return ToolResult("get_transactions", False, error=f"no account {account_id}")
    rows = TRANSACTIONS.get(account_id, [])
    if search:
        needle = search.lower()
        rows = [t for t in rows
                if needle in t["description"].lower() or needle in str(t["amount"])]
    return ToolResult("get_transactions", True,
                      data={"account_id": account_id, "matched": len(rows),
                            "transactions": rows})


def get_dispute_status(reference: str = "", **_) -> ToolResult:
    d = DISPUTES.get(reference.strip().upper())
    if not d:
        return ToolResult("get_dispute_status", False,
                          error=f"no dispute {reference or '(none supplied)'}")
    opened = date.fromisoformat(d["transaction_date"])
    return ToolResult("get_dispute_status", True,
                      data=dict(d, days_since_transaction=(date(2026, 9, 2) - opened).days))


# ------------------------------------------------------------------ write tool

def block_card(card_id: str = "", account_id: str = DEFAULT_ACCOUNT, **_) -> ToolResult:
    """Irreversible. The graph interrupts for human approval before calling this."""
    if not card_id:
        card_id = ACCOUNTS.get(account_id, {}).get("card_id", "")
    card = CARDS.get(card_id)
    if not card:
        return ToolResult("block_card", False, error=f"no card {card_id or '(none)'}")
    if card["status"] == "blocked":
        return ToolResult("block_card", True,
                          data=dict(card, note="already blocked, no change"))
    blocked = dict(card, status="blocked", blocked_on="2026-09-02")
    CARDS[card_id] = blocked
    return ToolResult("block_card", True, data=blocked)


REGISTRY = {
    "get_account_profile": get_account_profile,
    "get_transactions": get_transactions,
    "get_dispute_status": get_dispute_status,
    "block_card": block_card,
}

# Tools that change state. Approval is required before these run.
WRITE_TOOLS = {"block_card"}


def call(name: str, **kwargs) -> ToolResult:
    fn = REGISTRY.get(name)
    if fn is None:
        return ToolResult(name, False, error=f"unknown tool {name}")
    return fn(**kwargs)


def reset_fixtures() -> None:
    """Restore mutable fixture state. Used between tests."""
    for cid, card in CARDS.items():
        CARDS[cid] = {"card_id": cid, "account_id": card["account_id"], "status": "active"}
