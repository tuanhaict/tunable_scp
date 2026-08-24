from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..quantiles import conformal_quantile, validate_alpha_grid


@dataclass
class TsCPClassification:
    scores_d1: np.ndarray
    scores_d2: np.ndarray
    alpha_grid: np.ndarray
    delta: float

    def __post_init__(self) -> None:
        self.scores_d1 = np.asarray(self.scores_d1, dtype=float)
        self.scores_d2 = np.asarray(self.scores_d2, dtype=float)
        self.alpha_grid = validate_alpha_grid(self.alpha_grid)
        if self.delta < 0:
            raise ValueError("delta must be nonnegative.")

    def choose_alpha(self, label_scores: np.ndarray, budget: float, scores_d1: np.ndarray | None = None) -> float:
        target = float(budget) - self.delta
        if target < 0:
            raise ValueError("The theoretical construction requires S(x)-delta >= 0.")
        calibration = self.scores_d1 if scores_d1 is None else np.asarray(scores_d1, dtype=float)
        for alpha in self.alpha_grid:
            if np.sum(np.asarray(label_scores) <= conformal_quantile(calibration, alpha)) <= target:
                return float(alpha)
        return float(self.alpha_grid[-1])

    def predict_one(self, label_scores: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
        alpha = self.choose_alpha(label_scores, budget)
        return np.asarray(label_scores) <= conformal_quantile(self.scores_d2, alpha), alpha


@dataclass
class TsCPRegression:
    scores_d1: np.ndarray
    scores_d2: np.ndarray
    alpha_grid: np.ndarray
    delta: float

    def __post_init__(self) -> None:
        self.scores_d1 = np.asarray(self.scores_d1, dtype=float)
        self.scores_d2 = np.asarray(self.scores_d2, dtype=float)
        self.alpha_grid = validate_alpha_grid(self.alpha_grid)
        if self.delta < 0:
            raise ValueError("delta must be nonnegative.")

    def choose_alpha(self, scale: float, budget: float, scores_d1: np.ndarray | None = None) -> float:
        target = float(budget) - self.delta
        if target < 0:
            raise ValueError("The theoretical construction requires S(x)-delta >= 0.")
        calibration = self.scores_d1 if scores_d1 is None else np.asarray(scores_d1, dtype=float)
        for alpha in self.alpha_grid:
            if 2.0 * max(float(scale), 1e-12) * conformal_quantile(calibration, alpha) <= target:
                return float(alpha)
        return float(self.alpha_grid[-1])

    def predict_one(self, prediction: float, scale: float, budget: float) -> tuple[tuple[float, float], float]:
        alpha = self.choose_alpha(scale, budget)
        radius = max(float(scale), 1e-12) * conformal_quantile(self.scores_d2, alpha)
        return (float(prediction - radius), float(prediction + radius)), alpha

