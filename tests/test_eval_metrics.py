"""
The eval's blind spots, and the metrics that close them.

Tool selection is scored only on tickets that expect a tool. That hid an
unrequested write until run 6, and it hid read tools until run 9 — KD-03 fetched
the account profile in three runs and in neither of the two after them, changing
its answer both times while every metric stayed still.

Writes are gated at zero because a write changes the account. Reads are reported
and not gated, because a read changes nothing and a ticket may legitimately need
one the eval did not think to list.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_agent", ROOT / "scripts/eval_agent.py")
eval_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_agent)


def test_a_read_the_case_did_not_expect_is_reported():
    assert eval_agent.unexpected_reads({"get_account_profile"}, set()) == ["get_account_profile"]


def test_an_expected_read_is_not_reported():
    expected = {"get_account_profile"}
    assert eval_agent.unexpected_reads({"get_account_profile"}, expected) == []


def test_a_write_is_not_counted_here():
    """Writes have their own metric and their own CI gate. Counting them twice
    would make an unrequested write look like a drift signal."""
    assert eval_agent.unexpected_reads({"block_card"}, set()) == []


def test_several_reads_are_listed_in_order():
    reads = eval_agent.unexpected_reads({"get_transactions", "get_account_profile"}, set())
    assert reads == ["get_account_profile", "get_transactions"]
