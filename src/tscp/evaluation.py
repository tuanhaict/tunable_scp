from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .budgets import BudgetSpec, classification_uncertainty, evaluate_budget, normalized_uncertainty
from .data import DataSplit
from .methods.ecp import ECPClassification, ECPRegression
from .methods.tscp import TsCPClassification, TsCPRegression
from .models import ClassificationPredictions, RegressionPredictions
from .theory.coverage import CoverageEstimate, estimate_classification_coverage, estimate_regression_coverage


def classification_scores(probabilities: np.ndarray, score_type: str) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    if score_type == "one_minus_probability":
        return 1.0 - probs
    if score_type == "negative_log_probability":
        return -np.log(probs)
    raise ValueError(f"Unknown classification score {score_type!r}.")


@dataclass(frozen=True)
class Evaluation:
    covered: np.ndarray
    sizes: np.ndarray
    budgets: np.ndarray
    alphas: np.ndarray
    coverage_estimate: CoverageEstimate | None

    @property
    def coverage(self) -> float:
        return float(self.covered.mean())

    @property
    def average_size(self) -> float:
        return float(self.sizes.mean())

    @property
    def hard_constraint_accuracy(self) -> float:
        return float(np.mean(self.sizes <= self.budgets))


def _indices(pool_size: int, total: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if total % 2 or total > pool_size:
        raise ValueError("total calibration size must be even and no larger than the pool.")
    idx = np.random.default_rng(seed).choice(pool_size, total, replace=False)
    return idx[: total // 2], idx[total // 2 :]


def evaluate_regression(
    split: DataSplit, predictions: RegressionPredictions, method_name: str, total_calibration_size: int,
    budget_spec: BudgetSpec, delta: float, alpha_grid: np.ndarray, seed: int, number_test: int,
) -> Evaluation:
    d1, d2 = _indices(len(split.y_cal), total_calibration_size, seed)
    scores = np.abs(split.y_cal - predictions.pred_cal) / np.maximum(predictions.scale_cal, 1e-12)
    # The normalization reference is training-fitted and therefore independent of D1/D2 labels.
    u_cal = normalized_uncertainty(predictions.scale_cal, predictions.scale_reference)
    u_test = normalized_uncertainty(predictions.scale_test, predictions.scale_reference)
    budgets_cal = evaluate_budget(budget_spec, u_cal)
    budgets_test = evaluate_budget(budget_spec, u_test)
    limit = min(number_test, len(split.y_test))
    covered, sizes, alphas = [], [], []
    theory = None
    if method_name == "tscp":
        method = TsCPRegression(scores[d1], scores[d2], alpha_grid, delta)
        theory = estimate_regression_coverage(method, predictions.pred_cal[d1], predictions.scale_cal[d1], split.y_cal[d1], budgets_cal[d1])
    elif method_name == "ecp":
        method = ECPRegression(scores[np.concatenate([d1, d2])], alpha_grid)
    else:
        raise ValueError(f"Unknown adaptive method {method_name!r}.")
    for i in range(limit):
        interval, alpha = method.predict_one(predictions.pred_test[i], predictions.scale_test[i], budgets_test[i])
        covered.append(interval[0] <= split.y_test[i] <= interval[1])
        sizes.append(interval[1] - interval[0])
        alphas.append(alpha)
    return Evaluation(np.asarray(covered, float), np.asarray(sizes), budgets_test[:limit], np.asarray(alphas), theory)


def evaluate_classification(
    split: DataSplit, predictions: ClassificationPredictions, method_name: str, total_calibration_size: int,
    budget_spec: BudgetSpec, delta: float, alpha_grid: np.ndarray, seed: int, number_test: int, score_type: str,
    tie_break_epsilon: float = 0.0,
) -> Evaluation:
    d1, d2 = _indices(len(split.y_cal), total_calibration_size, seed)
    scores_cal = classification_scores(predictions.probs_cal, score_type)
    scores_test = classification_scores(predictions.probs_test, score_type)
    if tie_break_epsilon > 0:
        rng = np.random.default_rng(seed + 7919)
        scores_cal = scores_cal + rng.uniform(0.0, tie_break_epsilon, scores_cal.shape)
        scores_test = scores_test + rng.uniform(0.0, tie_break_epsilon, scores_test.shape)
    true_scores = scores_cal[np.arange(len(split.y_cal)), split.y_cal.astype(int)]
    uncertainty_kind = "entropy" if budget_spec.uncertainty == "auto" else budget_spec.uncertainty
    budgets_cal = evaluate_budget(budget_spec, classification_uncertainty(predictions.probs_cal, uncertainty_kind), classification=True)
    budgets_test = evaluate_budget(budget_spec, classification_uncertainty(predictions.probs_test, uncertainty_kind), classification=True)
    limit = min(number_test, len(split.y_test))
    covered, sizes, alphas = [], [], []
    theory = None
    if method_name == "tscp":
        method = TsCPClassification(true_scores[d1], true_scores[d2], alpha_grid, delta)
        theory = estimate_classification_coverage(method, scores_cal[d1], split.y_cal[d1], budgets_cal[d1])
    elif method_name == "ecp":
        method = ECPClassification(true_scores[np.concatenate([d1, d2])], alpha_grid)
    else:
        raise ValueError(f"Unknown adaptive method {method_name!r}.")
    for i in range(limit):
        prediction_set, alpha = method.predict_one(scores_test[i], budgets_test[i])
        covered.append(prediction_set[int(split.y_test[i])])
        sizes.append(prediction_set.sum())
        alphas.append(alpha)
    return Evaluation(np.asarray(covered, float), np.asarray(sizes, float), budgets_test[:limit], np.asarray(alphas), theory)
