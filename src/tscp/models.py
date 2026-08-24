from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import DataSplit


def _regressor(name: str, seed: int):
    key = name.lower()
    if key == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if key == "random_forest":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    if key == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed)
    if key == "extra_trees":
        return ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    if key in {"mlp", "dnn"}:
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300, random_state=seed))
    raise ValueError(f"Unknown regression model {name!r}.")


def _classifier(name: str, seed: int):
    key = name.lower()
    if key == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    if key == "random_forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if key == "gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    if key in {"mlp", "dnn"}:
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=seed))
    raise ValueError(f"Unknown classification model {name!r}.")


@dataclass(frozen=True)
class RegressionPredictions:
    pred_cal: np.ndarray
    scale_cal: np.ndarray
    pred_test: np.ndarray
    scale_test: np.ndarray
    scale_reference: np.ndarray


@dataclass(frozen=True)
class ClassificationPredictions:
    probs_cal: np.ndarray
    probs_test: np.ndarray


def fit_regression(split: DataSplit, mean_model: str, scale_model: str, seed: int) -> RegressionPredictions:
    x_mean, x_scale, y_mean, y_scale = train_test_split(split.x_train, split.y_train, test_size=0.5, random_state=seed)
    mean = _regressor(mean_model, seed)
    mean.fit(x_mean, y_mean)
    residuals = np.abs(y_scale - mean.predict(x_scale))
    scale = _regressor(scale_model, seed + 17)
    scale.fit(x_scale, residuals)
    raw_reference = np.maximum(scale.predict(x_scale), 1e-8)
    floor = max(float(np.quantile(raw_reference, 0.05)), 1e-8)
    predict_scale = lambda values: np.maximum(scale.predict(values), floor)
    return RegressionPredictions(
        mean.predict(split.x_cal),
        predict_scale(split.x_cal),
        mean.predict(split.x_test),
        predict_scale(split.x_test),
        np.maximum(raw_reference, floor),
    )


def fit_classification(split: DataSplit, model: str, seed: int) -> ClassificationPredictions:
    classifier = _classifier(model, seed)
    classifier.fit(split.x_train, split.y_train)
    return ClassificationPredictions(classifier.predict_proba(split.x_cal), classifier.predict_proba(split.x_test))
