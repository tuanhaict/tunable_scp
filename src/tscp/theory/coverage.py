from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..methods.tscp import TsCPClassification, TsCPRegression
from ..quantiles import conformal_quantile


@dataclass(frozen=True)
class CoverageEstimate:
    alpha_hat: float
    delta_hat: float
    old_proxy: float
    corrected_bound: float
    alpha_terms: np.ndarray
    delta_terms: np.ndarray


def _result(alpha_terms: list[float], delta_terms: list[float]) -> CoverageEstimate:
    alpha = np.asarray(alpha_terms, dtype=float)
    delta = np.asarray(delta_terms, dtype=float)
    alpha_hat = float(alpha.mean())
    delta_hat = float(delta.mean())
    return CoverageEstimate(alpha_hat, delta_hat, 1.0 - alpha_hat, 1.0 - alpha_hat - delta_hat, alpha, delta)


def _loo_quantiles(scores: np.ndarray, alpha_grid: np.ndarray) -> np.ndarray:
    """All leave-one-out conformal quantiles in O(n log n + n*|grid|)."""
    values = np.asarray(scores, dtype=float)
    grid = np.asarray(alpha_grid, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("LOO coverage estimation requires at least two D1 scores.")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    full_ranks = np.empty(n, dtype=int)
    full_ranks[order] = np.arange(n)
    output = np.empty((n, len(grid)), dtype=float)
    reduced_size = n - 1
    for j, alpha in enumerate(grid):
        rank = int(np.ceil((reduced_size + 1) * (1.0 - alpha)))
        if rank > reduced_size:
            output[:, j] = np.inf
            continue
        base = rank - 1
        indices = base + (full_ranks <= base)
        output[:, j] = sorted_values[indices]
    return output


def estimate_classification_coverage(
    method: TsCPClassification,
    d1_label_scores: np.ndarray,
    d1_true_labels: np.ndarray,
    d1_budgets: np.ndarray,
) -> CoverageEstimate:
    """Algorithm-2 LOO estimates of E[alpha_delta] and E[Delta_delta,n]."""
    all_scores = np.asarray(d1_label_scores, dtype=float)
    labels = np.asarray(d1_true_labels, dtype=int)
    budgets = np.asarray(d1_budgets, dtype=float)
    if len(all_scores) != len(method.scores_d1):
        raise ValueError("D1 candidate scores must align with D1 true-label scores.")
    loo_q1 = _loo_quantiles(method.scores_d1, method.alpha_grid)
    sizes = np.sum(all_scores[:, None, :] <= loo_q1[:, :, None], axis=2)
    feasible = sizes <= (budgets - method.delta)[:, None]
    first = np.argmax(feasible, axis=1)
    first[~np.any(feasible, axis=1)] = len(method.alpha_grid) - 1
    alpha = method.alpha_grid[first]
    q2 = method.q2_grid[first]
    misses = (all_scores[np.arange(len(all_scores)), labels] > q2).astype(float)
    alpha_terms = alpha.tolist()
    delta_terms = (misses - alpha).tolist()
    return _result(alpha_terms, delta_terms)


def estimate_regression_coverage(
    method: TsCPRegression,
    d1_predictions: np.ndarray,
    d1_scales: np.ndarray,
    d1_targets: np.ndarray,
    d1_budgets: np.ndarray,
) -> CoverageEstimate:
    predictions = np.asarray(d1_predictions, dtype=float)
    scales = np.asarray(d1_scales, dtype=float)
    targets = np.asarray(d1_targets, dtype=float)
    budgets = np.asarray(d1_budgets, dtype=float)
    if len(predictions) != len(method.scores_d1):
        raise ValueError("D1 observations must align with D1 scores.")
    true_scores = np.abs(targets - predictions) / np.maximum(scales, 1e-12)
    loo_q1 = _loo_quantiles(method.scores_d1, method.alpha_grid)
    feasible = 2.0 * np.maximum(scales, 1e-12)[:, None] * loo_q1 <= (budgets - method.delta)[:, None]
    first = np.argmax(feasible, axis=1)
    first[~np.any(feasible, axis=1)] = len(method.alpha_grid) - 1
    alpha = method.alpha_grid[first]
    q2 = method.q2_grid[first]
    misses = (true_scores > q2).astype(float)
    alpha_terms = alpha.tolist()
    delta_terms = (misses - alpha).tolist()
    return _result(alpha_terms, delta_terms)
