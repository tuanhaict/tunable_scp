import numpy as np
import pytest

from tscp.methods.ecp import ECPClassification, ECPRegression
from tscp.methods.tscp import TsCPClassification, TsCPRegression
from tscp.quantiles import conformal_quantile
from tscp.theory.coverage import (
    _loo_quantiles,
    estimate_classification_coverage,
    estimate_ecp_classification_alpha_loo,
    estimate_ecp_regression_alpha_loo,
    estimate_regression_coverage,
)


def test_fast_loo_quantiles_match_explicit_point_deletion():
    scores = np.array([0.4, 0.1, 0.4, 0.8, 0.2])
    grid = np.array([0.01, 0.2, 0.5, 0.8])
    expected = np.array([
        [conformal_quantile(np.delete(scores, i), alpha) for alpha in grid]
        for i in range(len(scores))
    ])
    np.testing.assert_allclose(_loo_quantiles(scores, grid), expected)


def test_ecp_regression_loo_matches_explicit_point_deletion():
    scores = np.array([0.2, 0.5, 0.8, 1.1])
    scales = np.array([0.8, 1.0, 1.2, 1.4])
    budgets = np.array([2.0, 2.0, 2.5, 3.0])
    grid = np.array([0.3, 0.5, 0.7, 0.9])
    expected = np.array([
        ECPRegression(np.delete(scores, i), grid).predict_one(0.0, scales[i], budgets[i])[1]
        for i in range(len(scores))
    ])
    actual = estimate_ecp_regression_alpha_loo(scores, scales, budgets, grid)
    np.testing.assert_allclose(actual, expected)


def test_ecp_classification_loo_matches_explicit_point_deletion():
    candidates = np.array([
        [0.1, 0.7, 0.9], [0.6, 0.2, 0.8], [0.7, 0.9, 0.3], [0.4, 0.5, 0.8],
    ])
    labels = np.array([0, 1, 2, 0])
    true_scores = candidates[np.arange(len(labels)), labels]
    budgets = np.array([2, 2, 1, 2])
    grid = np.array([0.2, 0.4, 0.6, 0.8])
    expected = np.array([
        ECPClassification(np.delete(true_scores, i), grid).predict_one(candidates[i], budgets[i])[1]
        for i in range(len(true_scores))
    ])
    actual = estimate_ecp_classification_alpha_loo(true_scores, candidates, budgets, grid)
    np.testing.assert_allclose(actual, expected)


def test_corrected_classification_bound_uses_delta_hat():
    method = TsCPClassification(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.15, 0.25, 0.35, 0.45]), np.array([0.2, 0.4, 0.6, 0.8]), 0.0)
    label_scores = np.array([[0.1, 0.8], [0.2, 0.7], [0.3, 0.6], [0.4, 0.5]])
    estimate = estimate_classification_coverage(method, label_scores, np.array([0, 0, 0, 0]), np.full(4, 2.0))
    assert estimate.corrected_bound == pytest.approx(1.0 - estimate.alpha_hat - estimate.delta_hat)
    assert len(estimate.delta_terms) == 4


def test_corrected_regression_bound_uses_delta_hat():
    method = TsCPRegression(np.array([0.2, 0.4, 0.6]), np.array([0.3, 0.5, 0.7]), np.array([0.2, 0.4, 0.6, 0.8]), 0.0)
    estimate = estimate_regression_coverage(method, np.zeros(3), np.ones(3), np.array([0.2, 0.4, 0.6]), np.full(3, 4.0))
    assert estimate.corrected_bound == pytest.approx(1.0 - estimate.alpha_hat - estimate.delta_hat)
