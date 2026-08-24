from __future__ import annotations

import numpy as np

from ..quantiles import conformal_quantile


def split_conformal_classification(label_scores: np.ndarray, calibration_scores: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(label_scores, dtype=float) <= conformal_quantile(calibration_scores, alpha)


def split_conformal_regression(prediction: float, scale: float, calibration_scores: np.ndarray, alpha: float) -> tuple[float, float]:
    q = conformal_quantile(calibration_scores, alpha)
    radius = max(float(scale), 1e-12) * q
    return float(prediction - radius), float(prediction + radius)

