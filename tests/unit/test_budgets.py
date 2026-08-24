import numpy as np
import pytest

from tscp.budgets import BudgetSpec, evaluate_budget


@pytest.mark.parametrize("kind", ["linear", "quadratic", "exponential"])
def test_adaptive_budgets_are_bounded_and_monotone(kind):
    spec = BudgetSpec(kind=kind, minimum=2.0, maximum=8.0, beta=2.0)
    values = evaluate_budget(spec, np.array([0.0, 0.25, 0.5, 1.0]))
    assert values[0] == 2.0
    assert values[-1] == pytest.approx(8.0)
    assert np.all(np.diff(values) >= 0)


def test_classification_budget_is_integer_valued():
    spec = BudgetSpec(kind="linear", minimum=1, maximum=5)
    values = evaluate_budget(spec, np.array([0.1, 0.6]), classification=True)
    assert np.all(values == np.ceil(values))

