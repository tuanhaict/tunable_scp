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
    alpha_terms: list[float] = []
    delta_terms: list[float] = []
    for k in range(len(all_scores)):
        reduced = np.delete(method.scores_d1, k)
        alpha_k = method.choose_alpha(all_scores[k], budgets[k], reduced)
        miss = float(all_scores[k, labels[k]] > conformal_quantile(method.scores_d2, alpha_k))
        alpha_terms.append(alpha_k)
        delta_terms.append(miss - alpha_k)
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
    alpha_terms: list[float] = []
    delta_terms: list[float] = []
    for k in range(len(predictions)):
        reduced = np.delete(method.scores_d1, k)
        alpha_k = method.choose_alpha(scales[k], budgets[k], reduced)
        miss = float(true_scores[k] > conformal_quantile(method.scores_d2, alpha_k))
        alpha_terms.append(alpha_k)
        delta_terms.append(miss - alpha_k)
    return _result(alpha_terms, delta_terms)
