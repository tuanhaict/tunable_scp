from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..quantiles import validate_alpha_grid


@dataclass
class ECPClassification:
    calibration_scores: np.ndarray
    alpha_grid: np.ndarray

    def __post_init__(self) -> None:
        self.calibration_scores = np.asarray(self.calibration_scores, dtype=float)
        self.alpha_grid = validate_alpha_grid(self.alpha_grid)

    def predict_one(self, label_scores: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
        scores = np.asarray(label_scores, dtype=float)
        denominator = (self.calibration_scores.sum() + scores) / (len(self.calibration_scores) + 1)
        e_values = scores / np.maximum(denominator, 1e-12)
        for alpha in self.alpha_grid:
            prediction_set = e_values < 1.0 / alpha
            if prediction_set.sum() <= budget:
                return prediction_set, float(alpha)
        return e_values < 1.0 / self.alpha_grid[-1], float(self.alpha_grid[-1])


@dataclass
class ECPRegression:
    calibration_scores: np.ndarray
    alpha_grid: np.ndarray

    def __post_init__(self) -> None:
        self.calibration_scores = np.asarray(self.calibration_scores, dtype=float)
        self.alpha_grid = validate_alpha_grid(self.alpha_grid)

    def predict_one(self, prediction: float, scale: float, budget: float) -> tuple[tuple[float, float], float]:
        total = self.calibration_scores.sum()
        n = len(self.calibration_scores)
        chosen_alpha = float(self.alpha_grid[-1])
        radius_score = float("inf")
        for alpha in self.alpha_grid:
            denominator = alpha * (n + 1) - 1.0
            current = float("inf") if denominator <= 0 else total / denominator
            if 2.0 * max(float(scale), 1e-12) * current <= budget:
                chosen_alpha, radius_score = float(alpha), float(current)
                break
        if not np.isfinite(radius_score):
            denominator = chosen_alpha * (n + 1) - 1.0
            radius_score = float("inf") if denominator <= 0 else total / denominator
        radius = max(float(scale), 1e-12) * radius_score
        return (float(prediction - radius), float(prediction + radius)), chosen_alpha

