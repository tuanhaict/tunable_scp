from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BudgetSpec:
    kind: str = "constant"
    value: float | None = None
    minimum: float = 1.0
    maximum: float | None = None
    beta: float = 3.0
    uncertainty: str = "auto"

    @classmethod
    def from_dict(cls, data: dict | None) -> "BudgetSpec":
        data = dict(data or {})
        return cls(
            kind=str(data.get("type", data.get("kind", "constant"))).lower(),
            value=None if data.get("value") is None else float(data["value"]),
            minimum=float(data.get("minimum", data.get("min", 1.0))),
            maximum=None if data.get("maximum", data.get("max")) is None else float(data.get("maximum", data.get("max"))),
            beta=float(data.get("beta", 3.0)),
            uncertainty=str(data.get("uncertainty", "auto")).lower(),
        )

    def validate(self) -> None:
        allowed = {"constant", "linear", "quadratic", "exponential"}
        if self.kind not in allowed:
            raise ValueError(f"Unknown budget type {self.kind!r}; expected one of {sorted(allowed)}.")
        if self.kind == "constant":
            if self.value is None or self.value <= 0:
                raise ValueError("A positive budget.value is required for a constant budget.")
        elif self.maximum is None or not self.minimum > 0 or self.maximum < self.minimum:
            raise ValueError("Adaptive budgets require 0 < minimum <= maximum.")
        if self.kind == "exponential" and self.beta <= 0:
            raise ValueError("Exponential budget beta must be positive.")


def normalized_uncertainty(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    lo, hi = np.quantile(reference, [0.05, 0.95])
    return np.clip((values - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def classification_uncertainty(probabilities: np.ndarray, kind: str = "entropy") -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    if kind in {"auto", "entropy"}:
        return -np.sum(probs * np.log(probs), axis=1) / np.log(probs.shape[1])
    if kind == "margin":
        ordered = np.sort(probs, axis=1)
        return 1.0 - (ordered[:, -1] - ordered[:, -2])
    raise ValueError(f"Unknown classification uncertainty {kind!r}.")


def evaluate_budget(spec: BudgetSpec, uncertainty: np.ndarray, *, classification: bool = False) -> np.ndarray:
    spec.validate()
    u = np.clip(np.asarray(uncertainty, dtype=float), 0.0, 1.0)
    if spec.kind == "constant":
        result = np.full(u.shape, float(spec.value))
    else:
        if spec.kind == "linear":
            transformed = u
        elif spec.kind == "quadratic":
            transformed = u**2
        else:
            transformed = np.expm1(spec.beta * u) / np.expm1(spec.beta)
        result = spec.minimum + (float(spec.maximum) - spec.minimum) * transformed
    if classification:
        result = np.ceil(result)
    return result.astype(float)

