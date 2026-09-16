"""
The spend cap is only as good as the price it multiplies by.

A run with the verifier on gpt-4o and everything else on gpt-4o-mini pays two
rates about 17x apart. Pricing all of it at one rate would let --max-cost pass
while the real spend ran past it.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_agent", ROOT / "scripts/eval_agent.py")
eval_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_agent)

PRICES = eval_agent.PRICES


def test_each_model_is_priced_at_its_own_rate():
    cost = eval_agent.run_cost({"gpt-4o-mini": [1_000_000, 0], "gpt-4o": [1_000_000, 0]})
    assert cost == pytest.approx(PRICES["gpt-4o-mini"][0] + PRICES["gpt-4o"][0])


def test_output_tokens_use_the_output_rate():
    assert eval_agent.run_cost({"gpt-4o": [0, 1_000_000]}) == pytest.approx(PRICES["gpt-4o"][1])


def test_an_unpriced_model_makes_the_whole_cost_unknown():
    """A sum that skipped the unpriced model would read as the full cost."""
    assert eval_agent.run_cost({"gpt-4o-mini": [10, 10], "some-new-model": [10, 10]}) is None
