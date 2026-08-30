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
        self.q1_grid = np.asarray([conformal_quantile(self.scores_d1, a) for a in self.alpha_grid])
        self.q2_grid = np.asarray([conformal_quantile(self.scores_d2, a) for a in self.alpha_grid])

    def choose_alpha(self, label_scores: np.ndarray, budget: float, scores_d1: np.ndarray | None = None) -> float:
        target = float(budget) - self.delta
        if target < 0:
            raise ValueError("The theoretical construction requires S(x)-delta >= 0.")
        quantiles = self.q1_grid if scores_d1 is None else np.asarray(
            [conformal_quantile(np.asarray(scores_d1, dtype=float), a) for a in self.alpha_grid]
        )
        sizes = np.sum(np.asarray(label_scores)[None, :] <= quantiles[:, None], axis=1)
        feasible = np.flatnonzero(sizes <= target)
        if feasible.size:
            return float(self.alpha_grid[feasible[0]])
        return float(self.alpha_grid[-1])

    def predict_one(self, label_scores: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
        alpha = self.choose_alpha(label_scores, budget)
        index = int(np.searchsorted(self.alpha_grid, alpha))
        return np.asarray(label_scores) <= self.q2_grid[index], alpha


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
        self.q1_grid = np.asarray([conformal_quantile(self.scores_d1, a) for a in self.alpha_grid])
        self.q2_grid = np.asarray([conformal_quantile(self.scores_d2, a) for a in self.alpha_grid])

    def choose_alpha(self, scale: float, budget: float, scores_d1: np.ndarray | None = None) -> float:
        target = float(budget) - self.delta
        if target < 0:
            raise ValueError("The theoretical construction requires S(x)-delta >= 0.")
        quantiles = self.q1_grid if scores_d1 is None else np.asarray(
            [conformal_quantile(np.asarray(scores_d1, dtype=float), a) for a in self.alpha_grid]
        )
        feasible = np.flatnonzero(2.0 * max(float(scale), 1e-12) * quantiles <= target)
        if feasible.size:
            return float(self.alpha_grid[feasible[0]])
        return float(self.alpha_grid[-1])

    def predict_one(self, prediction: float, scale: float, budget: float) -> tuple[tuple[float, float], float]:
        alpha = self.choose_alpha(scale, budget)
        index = int(np.searchsorted(self.alpha_grid, alpha))
        radius = max(float(scale), 1e-12) * self.q2_grid[index]
        return (float(prediction - radius), float(prediction + radius)), alpha
