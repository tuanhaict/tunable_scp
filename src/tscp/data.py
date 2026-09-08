from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_dataset(name: str, seed: int = 0, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    key = name.lower()
    if key == "synthetic_regression":
        x, y = make_regression(n_samples=max_samples or 20000, n_features=20, n_informative=12, noise=3.0, random_state=seed)
        return x, y * 0.05, "regression"
    if key == "california_housing":
        data = fetch_california_housing()
        x, y, task = data.data, data.target, "regression"
    elif key == "superconductivity":
        data = fetch_openml(data_id=43174, as_frame=False)
        x, y, task = data.data.astype(float), data.target.astype(float), "regression"
    elif key == "synthetic_classification":
        x, y = make_classification(n_samples=max_samples or 20000, n_features=20, n_informative=16, n_redundant=2, n_classes=20, random_state=seed)
        return x, y, "classification"
    elif key == "covertype":
        data = fetch_covtype()
        x, y, task = data.data, data.target - 1, "classification"
    elif key == "mnist":
        data = fetch_openml("mnist_784", version=1, as_frame=False)
        x, y, task = data.data / 255.0, LabelEncoder().fit_transform(data.target), "classification"
    else:
        raise ValueError(f"Unknown dataset {name!r}.")
    if max_samples is not None and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), max_samples, replace=False)
        x, y = x[idx], y[idx]
    return np.asarray(x), np.asarray(y), task


def split_dataset(x: np.ndarray, y: np.ndarray, task: str, seed: int, test_fraction: float = 0.05, calibration_fraction: float = 0.2) -> DataSplit:
    stratify = y if task == "classification" else None
    x_rest, x_test, y_rest, y_test = train_test_split(x, y, test_size=test_fraction, random_state=seed, stratify=stratify)
    relative_cal = calibration_fraction / (1.0 - test_fraction)
    stratify_rest = y_rest if task == "classification" else None
    x_train, x_cal, y_train, y_cal = train_test_split(x_rest, y_rest, test_size=relative_cal, random_state=seed + 1, stratify=stratify_rest)
    return DataSplit(x_train, y_train, x_cal, y_cal, x_test, y_test)

