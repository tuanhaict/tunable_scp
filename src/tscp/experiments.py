from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .budgets import BudgetSpec, classification_uncertainty, evaluate_budget, normalized_uncertainty
from .data import load_dataset, split_dataset
from .evaluation import classification_scores, evaluate_classification, evaluate_regression
from .methods.ecp import ECPClassification, ECPRegression
from .methods.tscp import TsCPClassification, TsCPRegression
from .models import fit_classification, fit_regression
from .quantiles import conformal_quantile


def alpha_grid(config: dict) -> np.ndarray:
    value = config.get("method", {}).get("alpha_grid", {})
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    return np.linspace(float(value.get("start", 0.01)), float(value.get("stop", 0.99)), int(value.get("steps", 99)))


def budget_for_dataset(config: dict, dataset: str, kind: str | None = None, value: float | None = None) -> BudgetSpec:
    raw = dict(config.get("budget", {}))
    by_dataset = raw.pop("value_by_dataset", {})
    if value is not None:
        raw["value"] = value
    elif dataset in by_dataset:
        raw["value"] = by_dataset[dataset]
    if kind is not None:
        raw["type"] = kind
    return BudgetSpec.from_dict(raw)


def prepare(config: dict, dataset: str, seed: int, model_override: str | None = None):
    maximum = config.get("data", {}).get("max_samples")
    x, y, task = load_dataset(dataset, seed=seed, max_samples=maximum)
    split = split_dataset(x, y, task, seed)
    model = config.get("model", {})
    if task == "regression":
        name = model_override or model.get("mean", "random_forest")
        fitted = fit_regression(split, name, model.get("scale", name), seed)
    else:
        name = model_override or model.get("classifier", "logistic")
        fitted = fit_classification(split, name, seed)
    return task, split, fitted


def evaluate(config: dict, dataset: str, seed: int, *, method: str = "tscp", calibration_size: int | None = None,
             number_test: int | None = None, delta: float | None = None, budget: BudgetSpec | None = None,
             score_type: str | None = None, model_override: str | None = None):
    task, split, fitted = prepare(config, dataset, seed, model_override)
    data_cfg, method_cfg = config.get("data", {}), config.get("method", {})
    calibration_size = int(calibration_size or data_cfg.get("total_calibration_size", 1000))
    number_test = int(number_test or data_cfg.get("fixed_number_test_samples", 1000))
    delta = float(method_cfg.get("delta", 0.1) if delta is None else delta)
    budget = budget or budget_for_dataset(config, dataset)
    grid = alpha_grid(config)
    if task == "regression":
        return task, evaluate_regression(split, fitted, method, calibration_size, budget, delta, grid, seed + 1000, number_test)
    score = score_type or method_cfg.get("score", "one_minus_probability")
    return task, evaluate_classification(
        split, fitted, method, calibration_size, budget, delta, grid, seed + 1000,
        number_test, score, float(method_cfg.get("tie_break_epsilon", 0.0)),
    )


def _theory_columns(result) -> dict:
    estimate = result.coverage_estimate
    return {
        "alpha_hat": np.nan if estimate is None else estimate.alpha_hat,
        "delta_hat": np.nan if estimate is None else estimate.delta_hat,
        "old_proxy": np.nan if estimate is None else estimate.old_proxy,
        "corrected_bound": np.nan if estimate is None else estimate.corrected_bound,
    }


def collect_self_validation(config: dict) -> pd.DataFrame:
    rows = []
    test_sizes = config["data"]["number_test_samples"]
    cal_sizes = config["data"]["total_calibration_sizes"]
    fixed_cal = int(config["data"]["total_calibration_size"])
    fixed_test = int(config["data"]["fixed_number_test_samples"])
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            _, maximum = evaluate(config, dataset, seed, calibration_size=fixed_cal, number_test=max(test_sizes))
            for count in test_sizes:
                count = min(int(count), len(maximum.covered))
                rows.append({"panel": "coverage", "dataset": dataset, "seed": seed, "x": count,
                             "empirical": float(maximum.covered[:count].mean()), "average_size": float(maximum.sizes[:count].mean()),
                             **_theory_columns(maximum)})
            for size in cal_sizes:
                _, result = evaluate(config, dataset, seed, calibration_size=int(size), number_test=fixed_test)
                rows.append({"panel": "size", "dataset": dataset, "seed": seed, "x": int(size),
                             "empirical": result.coverage, "average_size": result.average_size,
                             "budget": float(result.budgets.mean()), "hard_accuracy": result.hard_constraint_accuracy,
                             **_theory_columns(result)})
    return pd.DataFrame(rows)


def collect_delta_ablation(config: dict) -> pd.DataFrame:
    rows = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            for delta in config["experiment"]["deltas"]:
                _, result = evaluate(config, dataset, seed, delta=float(delta))
                rows.append({"dataset": dataset, "seed": seed, "delta": delta, "coverage": result.coverage,
                             "average_size": result.average_size, "budget": float(result.budgets.mean()),
                             "hard_accuracy": result.hard_constraint_accuracy, **_theory_columns(result)})
    return pd.DataFrame(rows)


def collect_compare(config: dict) -> pd.DataFrame:
    rows = []
    variants = config["experiment"].get("variants", [
        {"name": "TsCP", "method": "tscp", "delta": config["method"].get("delta", 0.1)},
        {"name": "eCP", "method": "ecp"},
    ])
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            for budget_value in config["experiment"]["budget_values"][dataset]:
                for variant in variants:
                    spec = budget_for_dataset(config, dataset, value=float(budget_value))
                    _, result = evaluate(config, dataset, seed, method=variant["method"], budget=spec,
                                         delta=variant.get("delta"), score_type=variant.get("score"))
                    rows.append({"dataset": dataset, "seed": seed, "budget": budget_value, "variant": variant["name"],
                                 "method": variant["method"], "coverage": result.coverage, "average_size": result.average_size,
                                 "hard_accuracy": result.hard_constraint_accuracy, **_theory_columns(result)})
    return pd.DataFrame(rows)


def collect_hard_constraint(config: dict) -> pd.DataFrame:
    rows = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            for size in config["data"]["total_calibration_sizes"]:
                _, result = evaluate(config, dataset, seed, calibration_size=int(size))
                rows.append({"dataset": dataset, "seed": seed, "calibration_size": size,
                             "hard_accuracy": result.hard_constraint_accuracy})
    return pd.DataFrame(rows)


def collect_budget_ablation(config: dict) -> pd.DataFrame:
    rows = []
    for dataset in config["datasets"]:
        for kind in config["experiment"]["budget_types"]:
            spec = budget_for_dataset(config, dataset, kind=kind)
            for seed in config["seeds"]:
                for method in ("tscp", "ecp"):
                    _, result = evaluate(config, dataset, seed, method=method, budget=spec)
                    rows.append({"dataset": dataset, "seed": seed, "budget_type": kind, "method": method,
                                 "coverage": result.coverage, "average_size": result.average_size,
                                 "hard_accuracy": result.hard_constraint_accuracy, **_theory_columns(result)})
    return pd.DataFrame(rows)


def collect_model_ablation(config: dict) -> pd.DataFrame:
    rows = []
    for dataset in config["datasets"]:
        model_names = config["experiment"].get("models_by_dataset", {}).get(dataset, config["experiment"].get("models", []))
        for model in model_names:
            for seed in config["seeds"]:
                for size in config["data"]["total_calibration_sizes"]:
                    _, result = evaluate(config, dataset, seed, calibration_size=int(size), model_override=model)
                    rows.append({"dataset": dataset, "model": model, "seed": seed, "calibration_size": size,
                                 "coverage": result.coverage, "average_size": result.average_size,
                                 "hard_accuracy": result.hard_constraint_accuracy, **_theory_columns(result)})
    return pd.DataFrame(rows)


def collect_loo_validation(config: dict) -> pd.DataFrame:
    rows = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            for size in config["data"]["total_calibration_sizes"]:
                _, result = evaluate(config, dataset, seed, calibration_size=int(size))
                estimate = result.coverage_estimate
                target_alpha = float(result.alphas.mean())
                target_delta = float(np.mean((1.0 - result.covered) - result.alphas))
                rows.append({
                    "dataset": dataset, "seed": seed, "calibration_size": size,
                    "alpha_hat": estimate.alpha_hat, "alpha_target": target_alpha,
                    "alpha_abs_error": abs(estimate.alpha_hat - target_alpha),
                    "delta_hat": estimate.delta_hat, "delta_target": target_delta,
                    "delta_abs_error": abs(estimate.delta_hat - target_delta),
                })
    return pd.DataFrame(rows)


def collect_loo_histogram(config: dict) -> pd.DataFrame:
    """Compare independent-test coverage with the corrected LOO estimate.

    A model/data split is fitted once for each outer seed.  Each trial then draws
    fresh D1/D2 calibration subsets, evaluates the resulting predictor on a
    large test batch, and computes 1-alpha_hat_LOO-delta_hat_LOO from D1.
    """
    rows = []
    data_cfg = config.get("data", {})
    method_cfg = config.get("method", {})
    trials = int(config.get("experiment", {}).get("trials", 200))
    number_test = int(data_cfg.get("fixed_number_test_samples", 2000))
    grid = alpha_grid(config)
    delta = float(method_cfg.get("delta", 0.1))
    score_type = method_cfg.get("score", "one_minus_probability")
    tie_break_epsilon = float(method_cfg.get("tie_break_epsilon", 0.0))

    for dataset in config["datasets"]:
        budget = budget_for_dataset(config, dataset)
        for outer_seed in config["seeds"]:
            task, split, fitted = prepare(config, dataset, int(outer_seed))
            for size in data_cfg["total_calibration_sizes"]:
                size = int(size)
                for trial in range(trials):
                    # Disjoint deterministic streams for every outer seed/size/trial.
                    trial_seed = int(outer_seed) * 10_000_000 + size * 1_000 + trial
                    if task == "regression":
                        result = evaluate_regression(
                            split, fitted, "tscp", size, budget, delta, grid,
                            trial_seed, number_test,
                        )
                    else:
                        result = evaluate_classification(
                            split, fitted, "tscp", size, budget, delta, grid,
                            trial_seed, number_test, score_type, tie_break_epsilon,
                        )
                    estimate = result.coverage_estimate
                    # Independent-test counterparts of the two terms in the
                    # corrected identity.  For test point j,
                    # delta_j = 1{Y_j not in C(X_j)} - alpha_j.
                    test_alpha = float(np.mean(result.alphas))
                    test_delta = float(np.mean((1.0 - result.covered) - result.alphas))
                    test_corrected = 1.0 - test_alpha - test_delta
                    rows.append({
                        "dataset": dataset,
                        "outer_seed": int(outer_seed),
                        "trial": trial,
                        "calibration_size": size,
                        "number_test": len(result.covered),
                        # Retained as a direct numerical identity check:
                        # test_corrected == test_coverage up to roundoff.
                        "test_coverage": result.coverage,
                        "test_alpha": test_alpha,
                        "test_delta": test_delta,
                        "test_corrected": test_corrected,
                        "loo_coverage": estimate.corrected_bound,
                        "alpha_hat_loo": estimate.alpha_hat,
                        "delta_hat_loo": estimate.delta_hat,
                    })
    return pd.DataFrame(rows)


def collect_runtime(config: dict) -> pd.DataFrame:
    """Time only conformal inference; fitting and score construction are outside the clock."""
    rows = []
    methods = config["experiment"].get("methods", ["scp", "tscp", "ecp", "ecp_tpss"])
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            task, split, fitted = prepare(config, dataset, seed)
            total = int(config["data"].get("total_calibration_size", 2000))
            idx = np.random.default_rng(seed + 1000).choice(len(split.y_cal), total, replace=False)
            d1, d2 = idx[: total // 2], idx[total // 2 :]
            grid = alpha_grid(config)
            fixed_alpha = float(config["experiment"].get("fixed_alpha", 0.1))
            delta = float(config["method"].get("delta", 0.1))
            spec = budget_for_dataset(config, dataset)
            if task == "regression":
                scores = np.abs(split.y_cal - fitted.pred_cal) / np.maximum(fitted.scale_cal, 1e-12)
                q_scp = conformal_quantile(scores[idx], fixed_alpha)
                tscp = TsCPRegression(scores[d1], scores[d2], grid, delta)
                ecp_tpss = ECPRegression(scores[idx], grid)
                denom = fixed_alpha * (len(idx) + 1) - 1.0
                ecp_fixed_radius_score = float("inf") if denom <= 0 else scores[idx].sum() / denom
                test_budgets = evaluate_budget(spec, normalized_uncertainty(fitted.scale_test, fitted.scale_reference))
            else:
                score_type = config["method"].get("score", "one_minus_probability")
                cal_scores = classification_scores(fitted.probs_cal, score_type)
                test_scores = classification_scores(fitted.probs_test, score_type)
                true_scores = cal_scores[np.arange(len(split.y_cal)), split.y_cal.astype(int)]
                q_scp = conformal_quantile(true_scores[idx], fixed_alpha)
                tscp = TsCPClassification(true_scores[d1], true_scores[d2], grid, delta)
                ecp_tpss = ECPClassification(true_scores[idx], grid)
                test_budgets = evaluate_budget(spec, classification_uncertainty(fitted.probs_test), classification=True)
            for count in config["data"]["number_test_samples"]:
                count = min(int(count), len(split.y_test))
                for label in methods:
                    start = time.perf_counter()
                    for i in range(count):
                        if task == "regression":
                            if label == "scp":
                                _ = 2.0 * fitted.scale_test[i] * q_scp
                            elif label == "tscp":
                                tscp.predict_one(fitted.pred_test[i], fitted.scale_test[i], test_budgets[i])
                            elif label == "ecp":
                                _ = 2.0 * fitted.scale_test[i] * ecp_fixed_radius_score
                            elif label == "ecp_tpss":
                                ecp_tpss.predict_one(fitted.pred_test[i], fitted.scale_test[i], test_budgets[i])
                            else:
                                raise ValueError(f"Unknown runtime method {label!r}.")
                        else:
                            if label == "scp":
                                _ = test_scores[i] <= q_scp
                            elif label == "tscp":
                                tscp.predict_one(test_scores[i], test_budgets[i])
                            elif label == "ecp":
                                denominator = (true_scores[idx].sum() + test_scores[i]) / (len(idx) + 1)
                                _ = test_scores[i] / np.maximum(denominator, 1e-12) < 1.0 / fixed_alpha
                            elif label == "ecp_tpss":
                                ecp_tpss.predict_one(test_scores[i], test_budgets[i])
                            else:
                                raise ValueError(f"Unknown runtime method {label!r}.")
                    elapsed = time.perf_counter() - start
                    rows.append({"dataset": dataset, "seed": seed, "number_test": count,
                                 "method": label, "runtime_seconds": elapsed})
    return pd.DataFrame(rows)


COLLECTORS = {
    "self_validation": collect_self_validation,
    "delta_ablation": collect_delta_ablation,
    "compare_ecp": collect_compare,
    "hard_constraint": collect_hard_constraint,
    "budget_ablation": collect_budget_ablation,
    "model_ablation": collect_model_ablation,
    "loo_validation": collect_loo_validation,
    "loo_histogram": collect_loo_histogram,
    "runtime": collect_runtime,
}


def _mean(frame: pd.DataFrame, group: list[str], columns: list[str]) -> pd.DataFrame:
    return frame.groupby(group, as_index=False)[columns].mean(numeric_only=True)


def make_figures(frame: pd.DataFrame, config: dict, output: Path) -> None:
    kind = config["experiment"]["type"]
    datasets = config["datasets"]
    if kind == "self_validation":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        cov = _mean(frame[frame.panel == "coverage"], ["dataset", "x"], ["empirical", "corrected_bound", "old_proxy"])
        size = _mean(frame[frame.panel == "size"], ["dataset", "x"], ["average_size", "budget"])
        for dataset in datasets:
            part = cov[cov.dataset == dataset]
            axes[0].plot(part.x, part.empirical, marker="o", label=f"{dataset} empirical")
            axes[0].plot(part.x, part.corrected_bound, linestyle="--", label=f"{dataset} corrected theory")
            part = size[size.dataset == dataset]
            axes[1].plot(part.x, part.average_size, marker="o", label=dataset)
            axes[1].plot(part.x, part.budget, linestyle=":", color=axes[1].lines[-1].get_color())
        axes[0].set(xlabel="Number of test samples", ylabel="Coverage")
        axes[1].set(xlabel=r"Total calibration size $2n$", ylabel="Average prediction-set size")
    elif kind == "delta_ablation":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        avg = _mean(frame, ["dataset", "delta"], ["coverage", "corrected_bound", "average_size", "budget"])
        for dataset in datasets:
            part = avg[avg.dataset == dataset]
            axes[0].plot(part.delta, part.coverage, marker="o", label=f"{dataset} empirical")
            axes[0].plot(part.delta, part.corrected_bound, linestyle="--", label=f"{dataset} corrected theory")
            axes[1].plot(part.delta, part.average_size, marker="o", label=dataset)
            axes[1].plot(part.delta, part.budget, linestyle=":", color=axes[1].lines[-1].get_color())
        axes[0].set(xlabel=r"Slack $\delta$", ylabel="Coverage")
        axes[1].set(xlabel=r"Slack $\delta$", ylabel="Average prediction-set size")
    elif kind in {"compare_ecp", "budget_ablation"}:
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
        key = "variant" if kind == "compare_ecp" else "budget_type"
        avg = _mean(frame, ["dataset", key, "method"] + (["budget"] if kind == "compare_ecp" else []), ["coverage", "average_size"])
        for ax, dataset in zip(axes[0], datasets):
            part = avg[avg.dataset == dataset]
            for labels, group in part.groupby([key, "method"]):
                if kind == "compare_ecp":
                    ax.plot(group.budget, group.coverage, marker="o", label=" / ".join(labels))
                else:
                    ax.plot(group.coverage, group.average_size, marker="o", label=" / ".join(labels))
            if kind == "compare_ecp":
                ax.set(title=dataset, xlabel="Pre-chosen set size", ylabel="Coverage")
            else:
                ax.set(title=dataset, xlabel="Coverage", ylabel="Average prediction-set size")
    elif kind == "model_ablation":
        fig, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 8), squeeze=False)
        avg = _mean(frame, ["dataset", "model", "calibration_size"], ["coverage", "average_size", "corrected_bound"])
        for col, dataset in enumerate(datasets):
            for model, part in avg[avg.dataset == dataset].groupby("model"):
                axes[0, col].plot(part.calibration_size, part.coverage, marker="o", label=model)
                axes[1, col].plot(part.calibration_size, part.average_size, marker="o", label=model)
            axes[0, col].set(title=dataset, xlabel=r"Total calibration size $2n$", ylabel="Coverage")
            axes[1, col].set(xlabel=r"Total calibration size $2n$", ylabel="Average prediction-set size")
    elif kind == "runtime":
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        avg = _mean(frame, ["dataset", "method", "number_test"], ["runtime_seconds"])
        for ax, dataset in zip(axes.flat, datasets):
            for method, part in avg[avg.dataset == dataset].groupby("method"):
                ax.plot(part.number_test, part.runtime_seconds, marker="o", label=method)
            ax.set(title=dataset, xlabel="Number of test samples", ylabel="Runtime (seconds)")
        for ax in axes.flat[len(datasets):]:
            ax.axis("off")
    elif kind == "loo_validation":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        avg = _mean(frame, ["dataset", "calibration_size"], ["alpha_abs_error", "delta_abs_error"])
        for dataset in datasets:
            part = avg[avg.dataset == dataset]
            axes[0].plot(part.calibration_size, part.alpha_abs_error, marker="o", label=dataset)
            axes[1].plot(part.calibration_size, part.delta_abs_error, marker="o", label=dataset)
        axes[0].set(xlabel=r"Total calibration size $2n$", ylabel=r"$|\hat\alpha-E[\alpha]|$")
        axes[1].set(xlabel=r"Total calibration size $2n$", ylabel=r"$|\hat\Delta-E[\Delta]|$")
    elif kind == "loo_histogram":
        calibration_sizes = [int(value) for value in config["data"]["total_calibration_sizes"]]
        fig, axes = plt.subplots(
            len(datasets), len(calibration_sizes),
            figsize=(4.2 * len(calibration_sizes), 3.0 * len(datasets)),
            squeeze=False, sharex=True,
        )
        bins = int(config.get("experiment", {}).get("bins", 20))
        # Compute the edges once for the complete grid.  Computing them inside
        # each panel makes equal-looking bars represent different intervals.
        all_values = np.concatenate([
            frame.test_corrected.to_numpy(dtype=float),
            frame.loo_coverage.to_numpy(dtype=float),
        ])
        all_values = all_values[np.isfinite(all_values)]
        if not len(all_values):
            raise ValueError("LOO histogram has no finite coverage values to plot.")
        configured_range = config.get("experiment", {}).get("histogram_range")
        histogram_range = None if configured_range is None else tuple(map(float, configured_range))
        histogram_bins = np.histogram_bin_edges(all_values, bins=bins, range=histogram_range)
        for row, dataset in enumerate(datasets):
            for col, size in enumerate(calibration_sizes):
                ax = axes[row, col]
                part = frame[(frame.dataset == dataset) & (frame.calibration_size == size)]
                show_legend = row == 0 and col == 0
                ax.hist(part.test_corrected, bins=histogram_bins, alpha=0.55,
                        label=r"$1-\hat\alpha_{\rm test}-\hat\delta_{\rm test}$" if show_legend else "_nolegend_",
                        color="tab:blue")
                ax.hist(part.loo_coverage, bins=histogram_bins, alpha=0.55,
                        label=r"$1-\hat\alpha^{\mathrm{LOO}}-\hat\delta^{\mathrm{LOO}}$" if show_legend else "_nolegend_",
                        color="tab:orange")
                ax.axvline(part.test_corrected.mean(), color="tab:blue", linestyle="--", linewidth=1.5)
                ax.axvline(part.loo_coverage.mean(), color="tab:orange", linestyle="--", linewidth=1.5)
                ax.set_xlim(histogram_bins[0], histogram_bins[-1])
                if row == 0:
                    ax.set_title(rf"$N_{{\mathrm{{cal}}}}=2n={size}$")
                if col == 0:
                    ax.set_ylabel(f"{dataset}\nFrequency")
                if row == len(datasets) - 1:
                    ax.set_xlabel("Coverage")
                if show_legend:
                    ax.legend(fontsize=7)
    else:
        return
    for ax in fig.axes:
        if ax.has_data():
            ax.grid(alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "figure.pdf", bbox_inches="tight")
    fig.savefig(output / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_hard_table(frame: pd.DataFrame, output: Path) -> None:
    table = frame.groupby(["calibration_size", "dataset"])["hard_accuracy"].mean().unstack("dataset")
    table.to_csv(output / "hard_constraint_table.csv")
    (output / "hard_constraint_table.tex").write_text(table.to_latex(float_format="%.3f"), encoding="utf-8")


def environment_metadata() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}
