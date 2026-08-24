from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the ceil((n+1)(1-alpha))-th conformal order statistic."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("At least one calibration score is required.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    rank = int(np.ceil((values.size + 1) * (1.0 - float(alpha))))
    if rank > values.size:
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])


def validate_alpha_grid(alpha_grid: np.ndarray) -> np.ndarray:
    grid = np.unique(np.asarray(alpha_grid, dtype=float))
    if grid.size == 0 or np.any(grid <= 0.0) or np.any(grid >= 1.0):
        raise ValueError("alpha_grid must be nonempty and contained in (0, 1).")
    return np.sort(grid)

