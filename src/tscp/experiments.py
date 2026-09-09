from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .budgets import BudgetSpec, classification_uncertainty, evaluate_budget, normalized_uncertainty
from .data import load_dataset, split_dataset
from .evaluation import classification_scores, evaluate_classification, evaluate_regression
from .methods.ecp import ECPClassification, ECPRegression
from .methods.tscp import TsCPClassification, TsCPRegression
from .models import fit_classification, fit_regression
from .quantiles import conformal_quantile
from .theory.coverage import estimate_ecp_classification_alpha_loo, estimate_ecp_regression_alpha_loo


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
    """Validate the corrected coverage identity with independent MC streams.

    The theoretical curve is estimated by an independent Monte Carlo stream.
    Every reference trial redraws the full calibration sample C=(D1,D2) and one
    test observation (X0,Y0), then records alpha_C(X0) and
    Delta_C(X0)=1{Y0 not in C_C(X0)}-alpha_C(X0).  In particular, this
    experiment does not use a leave-one-out coverage estimate.  The empirical
    stream independently redraws the same pair (C,(X0,Y0)) on every trial and
    reports prefix means at the configured numbers of test samples.
    """
    rows = []
    data_cfg = config["data"]
    method_cfg = config.get("method", {})
    reference_trials = int(config.get("experiment", {}).get("reference_trials", 500))
    if reference_trials < 1:
        raise ValueError("Self-validation requires at least one reference trial.")
    test_sizes = config["data"]["number_test_samples"]
    cal_sizes = config["data"]["total_calibration_sizes"]
    fixed_cal = int(config["data"]["total_calibration_size"])
    fixed_test = int(config["data"]["fixed_number_test_samples"])
    grid = alpha_grid(config)
    delta = float(method_cfg.get("delta", 0.1))
    score_type = method_cfg.get("score", "one_minus_probability")
    tie_break_epsilon = float(method_cfg.get("tie_break_epsilon", 0.0))

    for dataset in config["datasets"]:
        budget = budget_for_dataset(config, dataset)
        for seed in config["seeds"]:
            task, split, fitted = prepare(config, dataset, int(seed))

            def run_tscp(calibration_size: int, number_test: int, run_seed: int,
                         trial_split=split, trial_fitted=fitted):
                if task == "regression":
                    return evaluate_regression(
                        trial_split, trial_fitted, "tscp", calibration_size, budget,
                        delta, grid, run_seed, number_test, estimate_coverage=False,
                    )
                return evaluate_classification(
                    trial_split, trial_fitted, "tscp", calibration_size, budget,
                    delta, grid, run_seed, number_test, score_type,
                    tie_break_epsilon, estimate_coverage=False,
                )

            def run_one_random_pair(calibration_size: int, run_seed: int):
                """Redraw C=(D1,D2) and one independent test observation."""
                pair_rng = np.random.default_rng(run_seed)
                # evaluate_* initializes the same RNG and therefore redraws
                # exactly these calibration indices internally.
                pair_rng.choice(len(split.y_cal), calibration_size, replace=False)
                test_index = int(pair_rng.integers(0, len(split.y_test)))
                trial_split = replace(
                    split,
                    x_test=split.x_test[[test_index]],
                    y_test=split.y_test[[test_index]],
                )
                if task == "regression":
                    trial_fitted = replace(
                        fitted,
                        pred_test=fitted.pred_test[[test_index]],
                        scale_test=fitted.scale_test[[test_index]],
                    )
                else:
                    trial_fitted = replace(
                        fitted,
                        probs_test=fitted.probs_test[[test_index]],
                    )
                return run_tscp(
                    calibration_size, 1, run_seed, trial_split, trial_fitted,
                )

            # Marginal empirical coverage. Trial r redraws both C_r and X_0,r;
            # the configured test counts are nested prefixes of this stream.
            empirical_covered = []
            empirical_sizes = []
            for empirical_trial in range(max(map(int, test_sizes))):
                empirical_seed = (
                    2_000_000_000 + int(seed) * 10_000_000
                    + fixed_cal * 1_000 + empirical_trial
                )
                empirical_result = run_one_random_pair(fixed_cal, empirical_seed)
                empirical_covered.append(float(empirical_result.covered[0]))
                empirical_sizes.append(float(empirical_result.sizes[0]))
            empirical_covered = np.asarray(empirical_covered, dtype=float)
            empirical_sizes = np.asarray(empirical_sizes, dtype=float)

            # Independent Monte Carlo estimate of E[alpha_C(X0)] and
            # E[Delta_C(X0)] on a seed stream disjoint from the empirical one.
            reference_alphas = []
            reference_deltas = []
            reference_covered = []
            for reference_trial in range(reference_trials):
                reference_seed = (
                    1_000_000_000 + int(seed) * 10_000_000
                    + fixed_cal * 1_000 + reference_trial
                )
                reference = run_one_random_pair(fixed_cal, reference_seed)
                alpha_value = float(reference.alphas[0])
                covered_value = float(reference.covered[0])
                reference_alphas.append(alpha_value)
                reference_deltas.append((1.0 - covered_value) - alpha_value)
                reference_covered.append(covered_value)

            expected_alpha = float(np.mean(reference_alphas))
            expected_delta = float(np.mean(reference_deltas))
            reference_coverage = float(np.mean(reference_covered))
            corrected_theory = 1.0 - expected_alpha - expected_delta
            theory_columns = {
                "expected_alpha": expected_alpha,
                "expected_delta": expected_delta,
                "old_proxy": 1.0 - expected_alpha,
                "corrected_bound": corrected_theory,
                "reference_coverage": reference_coverage,
                "reference_trials": reference_trials,
            }

            for count in test_sizes:
                count = int(count)
                rows.append({"panel": "coverage", "dataset": dataset, "seed": seed, "x": count,
                             "empirical": float(empirical_covered[:count].mean()),
                             "average_size": float(empirical_sizes[:count].mean()),
                             "empirical_trials": count,
                             **theory_columns})
            for size in cal_sizes:
                result = run_tscp(int(size), fixed_test, int(seed) + 1000)
                rows.append({"panel": "size", "dataset": dataset, "seed": seed, "x": int(size),
                             "empirical": result.coverage, "average_size": result.average_size,
                             "budget": float(result.budgets.mean()), "hard_accuracy": result.hard_constraint_accuracy})
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


def collect_loo_compare_ecp(config: dict) -> pd.DataFrame:
    """Compare LOO estimators against independent Monte Carlo expectations."""
    rows = []
    data_cfg = config.get("data", {})
    method_cfg = config.get("method", {})
    experiment_cfg = config.get("experiment", {})
    trials = int(experiment_cfg.get("trials", 20))
    reference_trials = int(experiment_cfg.get("reference_trials", 500))
    if trials < 2:
        raise ValueError("LOO comparison requires at least two evaluation trials for variance.")
    if reference_trials < 2:
        raise ValueError("LOO comparison requires at least two independent reference trials.")
    grid = alpha_grid(config)
    delta = float(method_cfg.get("delta", 0.1))
    score_type = method_cfg.get("score", "one_minus_probability")
    ecp_score_type = experiment_cfg.get("ecp_score", score_type)
    tie_epsilon = float(method_cfg.get("tie_break_epsilon", 0.0))

    for dataset in config["datasets"]:
        budget = budget_for_dataset(config, dataset)
        for outer_seed in config["seeds"]:
            task, split, fitted = prepare(config, dataset, int(outer_seed))
            if task == "regression":
                all_scores = np.abs(split.y_cal - fitted.pred_cal) / np.maximum(fitted.scale_cal, 1e-12)
                cal_uncertainty = normalized_uncertainty(fitted.scale_cal, fitted.scale_reference)
                cal_budgets = evaluate_budget(budget, cal_uncertainty)
            else:
                uncertainty_kind = "entropy" if budget.uncertainty == "auto" else budget.uncertainty
                cal_budgets = evaluate_budget(
                    budget, classification_uncertainty(fitted.probs_cal, uncertainty_kind), classification=True,
                )

            def one_test_view(test_index: int):
                trial_split = replace(
                    split, x_test=split.x_test[[test_index]], y_test=split.y_test[[test_index]],
                )
                if task == "regression":
                    trial_fitted = replace(
                        fitted,
                        pred_test=fitted.pred_test[[test_index]],
                        scale_test=fitted.scale_test[[test_index]],
                    )
                else:
                    trial_fitted = replace(fitted, probs_test=fitted.probs_test[[test_index]])
                return trial_split, trial_fitted

            for size in data_cfg["total_calibration_sizes"]:
                size = int(size)

                # Estimate each target expectation using an independent seed
                # stream.  Every draw resamples calibration and one test point.
                reference = {"truncated_eCP": [], "TsCP": []}
                reference_alpha = {"truncated_eCP": [], "TsCP": []}
                reference_delta = {"truncated_eCP": [], "TsCP": []}
                reference_covered = {"truncated_eCP": [], "TsCP": []}
                for reference_trial in range(reference_trials):
                    reference_seed = (
                        1_000_000_000 + int(outer_seed) * 10_000_000
                        + size * 1_000 + reference_trial
                    )
                    reference_rng = np.random.default_rng(reference_seed)
                    reference_rng.choice(len(split.y_cal), size, replace=False)
                    test_index = int(reference_rng.integers(0, len(split.y_test)))
                    trial_split, trial_fitted = one_test_view(test_index)
                    if task == "regression":
                        tscp_reference = evaluate_regression(
                            trial_split, trial_fitted, "tscp", size, budget, delta,
                            grid, reference_seed, 1, estimate_coverage=False,
                        )
                        ecp_reference = evaluate_regression(
                            trial_split, trial_fitted, "ecp", size, budget, delta,
                            grid, reference_seed, 1, estimate_coverage=False,
                        )
                    else:
                        tscp_reference = evaluate_classification(
                            trial_split, trial_fitted, "tscp", size, budget, delta,
                            grid, reference_seed, 1, score_type, tie_epsilon,
                            estimate_coverage=False,
                        )
                        ecp_reference = evaluate_classification(
                            trial_split, trial_fitted, "ecp", size, budget, delta,
                            grid, reference_seed, 1, ecp_score_type, tie_epsilon,
                            estimate_coverage=False,
                        )
                    ecp_alpha = float(ecp_reference.alphas[0])
                    tscp_alpha = float(tscp_reference.alphas[0])
                    tscp_delta = float((1.0 - tscp_reference.covered[0]) - tscp_alpha)
                    reference_alpha["truncated_eCP"].append(ecp_alpha)
                    reference_delta["truncated_eCP"].append(0.0)
                    reference["truncated_eCP"].append(ecp_alpha)
                    reference_covered["truncated_eCP"].append(float(ecp_reference.covered[0]))
                    reference_alpha["TsCP"].append(tscp_alpha)
                    reference_delta["TsCP"].append(tscp_delta)
                    reference["TsCP"].append(tscp_alpha + tscp_delta)
                    reference_covered["TsCP"].append(float(tscp_reference.covered[0]))

                reference_stats = {}
                for method_name in ("truncated_eCP", "TsCP"):
                    samples = np.asarray(reference[method_name], dtype=float)
                    covered_samples = np.asarray(reference_covered[method_name], dtype=float)
                    reference_stats[method_name] = {
                        "alpha": float(np.mean(reference_alpha[method_name])),
                        "delta": float(np.mean(reference_delta[method_name])),
                        "target": float(samples.mean()),
                        "standard_error": float(samples.std(ddof=1) / np.sqrt(reference_trials)),
                        "empirical_coverage": float(covered_samples.mean()),
                        "coverage_standard_error": float(covered_samples.std(ddof=1) / np.sqrt(reference_trials)),
                    }

                # Independent evaluation stream: only these LOO estimates enter
                # the variance and absolute-error curves.
                for trial in range(trials):
                    trial_seed = int(outer_seed) * 10_000_000 + size * 1_000 + trial
                    trial_rng = np.random.default_rng(trial_seed)
                    selected = trial_rng.choice(len(split.y_cal), size, replace=False)
                    test_index = int(trial_rng.integers(0, len(split.y_test)))
                    trial_split, trial_fitted = one_test_view(test_index)
                    if task == "regression":
                        tscp_result = evaluate_regression(
                            trial_split, trial_fitted, "tscp", size, budget, delta,
                            grid, trial_seed, 1,
                        )
                        ecp_result = evaluate_regression(
                            trial_split, trial_fitted, "ecp", size, budget, delta,
                            grid, trial_seed, 1,
                        )
                        ecp_terms = estimate_ecp_regression_alpha_loo(
                            all_scores[selected], fitted.scale_cal[selected], cal_budgets[selected], grid,
                        )
                    else:
                        trial_fitted = replace(fitted, probs_test=fitted.probs_test[[test_index]])
                        cal_scores = classification_scores(fitted.probs_cal, ecp_score_type)
                        if tie_epsilon > 0:
                            rng = np.random.default_rng(trial_seed + 7919)
                            cal_scores = cal_scores + rng.uniform(0.0, tie_epsilon, cal_scores.shape)
                        true_scores = cal_scores[np.arange(len(split.y_cal)), split.y_cal.astype(int)]
                        tscp_result = evaluate_classification(
                            trial_split, trial_fitted, "tscp", size, budget, delta,
                            grid, trial_seed, 1, score_type, tie_epsilon,
                        )
                        ecp_result = evaluate_classification(
                            trial_split, trial_fitted, "ecp", size, budget, delta,
                            grid, trial_seed, 1, ecp_score_type, tie_epsilon,
                        )
                        ecp_terms = estimate_ecp_classification_alpha_loo(
                            true_scores[selected], cal_scores[selected], cal_budgets[selected], grid,
                        )

                    tscp_estimate = tscp_result.coverage_estimate
                    tscp_test_alpha = float(tscp_result.alphas[0])
                    tscp_test_delta = float((1.0 - tscp_result.covered[0]) - tscp_test_alpha)
                    values = [
                        ("truncated_eCP", float(ecp_terms.mean()), 0.0,
                         float(ecp_result.alphas[0]), 0.0),
                        ("TsCP", tscp_estimate.alpha_hat, tscp_estimate.delta_hat,
                         tscp_test_alpha, tscp_test_delta),
                    ]
                    for method_name, loo_alpha, loo_delta, test_alpha, test_delta in values:
                        estimate_value = loo_alpha + loo_delta
                        stats = reference_stats[method_name]
                        rows.append({
                            "dataset": dataset,
                            "outer_seed": int(outer_seed),
                            "trial": trial,
                            "test_index": test_index,
                            "number_test": 1,
                            "calibration_size": size,
                            "method": method_name,
                            "alpha_hat_loo": loo_alpha,
                            "delta_hat_loo": loo_delta,
                            "loo_estimate": estimate_value,
                            "test_alpha_sample": test_alpha,
                            "test_delta_sample": test_delta,
                            "test_target_sample": test_alpha + test_delta,
                            "reference_trials": reference_trials,
                            "reference_alpha": stats["alpha"],
                            "reference_delta": stats["delta"],
                            "reference_target": stats["target"],
                            "reference_standard_error": stats["standard_error"],
                            "coverage_estimate": 1.0 - estimate_value,
                            "reference_empirical_coverage": stats["empirical_coverage"],
                            "reference_coverage_standard_error": stats["coverage_standard_error"],
                            "absolute_error": abs(estimate_value - stats["target"]),
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
    "loo_compare_ecp": collect_loo_compare_ecp,
    "runtime": collect_runtime,
}


def _mean(frame: pd.DataFrame, group: list[str], columns: list[str]) -> pd.DataFrame:
    return frame.groupby(group, as_index=False)[columns].mean(numeric_only=True)


_DATASET_DISPLAY_NAMES = {
    "synthetic_regression": "SyntheticRegression",
    "syntheticregression": "SyntheticRegression",
    "california_housing": "CaliforniaHousing",
    "californiahousing": "CaliforniaHousing",
    "superconductivity": "Superconductivity",
    "synthetic_classification": "SyntheticClassification",
    "syntheticclassification": "SyntheticClassification",
    "mnist": "MNIST",
    "fashion_mnist": "FashionMNIST",
    "covertype": "Covertype",
    "adult": "Adult",
    "electricity": "Electricity",
    "year": "Year",
    "diamonds": "Diamonds",
    "allstate": "Allstate",
    "minibone": "MiniBooNE",
}


def _dataset_display_name(dataset: str) -> str:
    """Return the publication-facing name while preserving raw data keys."""
    key = str(dataset)
    return _DATASET_DISPLAY_NAMES.get(
        key.lower(), "".join(part.capitalize() for part in key.split("_")),
    )


def _plot_figsize(config: dict, default: tuple[float, float]) -> tuple[float, float]:
    value = config.get("plot", {}).get("figsize")
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("plot.figsize must contain [width, height].")
    width, height = map(float, value)
    if width <= 0 or height <= 0:
        raise ValueError("Figure width and height must be positive.")
    return width, height


def _apply_figure_style(fig, config: dict) -> None:
    """Apply YAML/CLI figure options after all artists have been created."""
    plot = config.get("plot", {})
    base_font = plot.get("font_size")

    def font_value(key: str):
        value = plot.get(key)
        return base_font if value is None else value

    title_font = font_value("title_font_size")
    label_font = font_value("label_font_size")
    tick_font = font_value("tick_font_size")
    legend_font = font_value("legend_font_size")
    line_width = plot.get("line_width")
    marker_size = plot.get("marker_size")
    max_x_ticks = plot.get("max_x_ticks")
    max_y_ticks = plot.get("max_y_ticks")

    def axis_limits(value, name: str):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"plot.{name} must contain [minimum, maximum].")
        lower, upper = map(float, value)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"plot.{name} must contain two finite values with minimum < maximum.")
        return lower, upper

    global_x_limits = axis_limits(plot.get("xlim"), "xlim")
    global_y_limits = axis_limits(plot.get("ylim"), "ylim")
    scoped_limits = {
        "coverage": (
            axis_limits(plot.get("coverage_xlim"), "coverage_xlim"),
            axis_limits(plot.get("coverage_ylim"), "coverage_ylim"),
        ),
        "size": (
            axis_limits(plot.get("size_xlim"), "size_xlim"),
            axis_limits(plot.get("size_ylim"), "size_ylim"),
        ),
    }

    for ax in fig.axes:
        scope = getattr(ax, "_tscp_plot_scope", None)
        scope_x, scope_y = scoped_limits.get(scope, (None, None))
        x_limits = scope_x if scope_x is not None else global_x_limits
        y_limits = scope_y if scope_y is not None else global_y_limits
        if title_font is not None:
            ax.title.set_fontsize(float(title_font))
        if label_font is not None:
            ax.xaxis.label.set_fontsize(float(label_font))
            ax.yaxis.label.set_fontsize(float(label_font))
        if tick_font is not None:
            ax.tick_params(axis="both", labelsize=float(tick_font))
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        if max_x_ticks is not None and ax.get_xscale() == "linear":
            ax.xaxis.set_major_locator(MaxNLocator(nbins=int(max_x_ticks)))
        if max_y_ticks is not None and ax.get_yscale() == "linear":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=int(max_y_ticks)))
        if line_width is not None:
            for line in ax.lines:
                line.set_linewidth(float(line_width))
        if marker_size is not None:
            for line in ax.lines:
                if line.get_marker() not in (None, "None", "", " "):
                    line.set_markersize(float(marker_size))
            for collection in ax.collections:
                if isinstance(collection, PathCollection):
                    collection.set_sizes([float(marker_size) ** 2])
        legend = ax.get_legend()
        if legend is not None and legend_font is not None:
            for text_item in legend.get_texts():
                text_item.set_fontsize(float(legend_font))
            if legend.get_title() is not None:
                legend.get_title().set_fontsize(float(legend_font))


def _save_figure(fig, output: Path, stems: str | tuple[str, ...], config: dict) -> None:
    _apply_figure_style(fig, config)
    fig.tight_layout()
    dpi = int(config.get("plot", {}).get("dpi", 180))
    if dpi <= 0:
        raise ValueError("plot.dpi must be positive.")
    names = (stems,) if isinstance(stems, str) else stems
    for stem in names:
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def summarize_loo_compare(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the per-seed and plotted point tables for the LOO comparison."""
    groups = ["dataset", "method", "calibration_size", "outer_seed"]
    per_seed = frame.groupby(groups, as_index=False).agg(
        evaluation_trials=("trial", "nunique"),
        estimator_variance=("loo_estimate", "var"),
        mean_absolute_error=("absolute_error", "mean"),
        reference_trials=("reference_trials", "first"),
        reference_alpha=("reference_alpha", "first"),
        reference_delta=("reference_delta", "first"),
        reference_target=("reference_target", "first"),
        reference_standard_error=("reference_standard_error", "first"),
        mean_coverage_estimate=("coverage_estimate", "mean"),
        empirical_coverage=("reference_empirical_coverage", "first"),
        empirical_coverage_standard_error=("reference_coverage_standard_error", "first"),
    )
    per_seed["coverage_gap"] = np.abs(
        per_seed["mean_coverage_estimate"] - per_seed["empirical_coverage"]
    )
    summary = per_seed.groupby(
        ["dataset", "method", "calibration_size"], as_index=False,
    ).agg(
        outer_seeds=("outer_seed", "nunique"),
        evaluation_trials=("evaluation_trials", "first"),
        reference_trials=("reference_trials", "first"),
        variance_mean=("estimator_variance", "mean"),
        variance_std=("estimator_variance", "std"),
        absolute_error_mean=("mean_absolute_error", "mean"),
        absolute_error_std=("mean_absolute_error", "std"),
        reference_alpha_mean=("reference_alpha", "mean"),
        reference_delta_mean=("reference_delta", "mean"),
        reference_target_mean=("reference_target", "mean"),
        reference_target_std=("reference_target", "std"),
        reference_mc_standard_error_mean=("reference_standard_error", "mean"),
        coverage_estimate_mean=("mean_coverage_estimate", "mean"),
        empirical_coverage_mean=("empirical_coverage", "mean"),
        coverage_gap_mean=("coverage_gap", "mean"),
        coverage_gap_std=("coverage_gap", "std"),
        empirical_coverage_mc_standard_error_mean=("empirical_coverage_standard_error", "mean"),
    )
    for column in ("variance_std", "absolute_error_std", "reference_target_std", "coverage_gap_std"):
        summary[column] = summary[column].fillna(0.0)
    return per_seed, summary


def loo_compare_report_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compact wide table containing only the points shown on the two plots."""
    table = summary.pivot(
        index=["dataset", "calibration_size"],
        columns="method",
        values=["variance_mean", "absolute_error_mean"],
    )
    table.columns = [f"{metric}__{method}" for metric, method in table.columns]
    table = table.reset_index().rename(columns={
        "calibration_size": "total_calibration_size",
        "variance_mean__truncated_eCP": "ecp_variance",
        "variance_mean__TsCP": "tscp_variance",
        "absolute_error_mean__truncated_eCP": "ecp_absolute_error",
        "absolute_error_mean__TsCP": "tscp_absolute_error",
    })
    columns = [
        "dataset", "total_calibration_size",
        "ecp_variance", "tscp_variance",
        "ecp_absolute_error", "tscp_absolute_error",
    ]
    return table[columns].sort_values(["dataset", "total_calibration_size"]).reset_index(drop=True)


def loo_coverage_report_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compact table for estimated-vs-empirical coverage and their gaps."""
    table = summary.pivot(
        index=["dataset", "calibration_size"], columns="method",
        values=[
            "coverage_estimate_mean", "empirical_coverage_mean",
            "coverage_gap_mean", "coverage_gap_std",
        ],
    )
    table.columns = [f"{metric}__{method}" for metric, method in table.columns]
    table = table.reset_index().rename(columns={
        "calibration_size": "total_calibration_size",
        "coverage_estimate_mean__truncated_eCP": "ecp_estimated_coverage",
        "empirical_coverage_mean__truncated_eCP": "ecp_empirical_coverage",
        "coverage_gap_mean__truncated_eCP": "ecp_coverage_gap",
        "coverage_gap_std__truncated_eCP": "ecp_coverage_gap_std",
        "coverage_estimate_mean__TsCP": "tscp_estimated_coverage",
        "empirical_coverage_mean__TsCP": "tscp_empirical_coverage",
        "coverage_gap_mean__TsCP": "tscp_coverage_gap",
        "coverage_gap_std__TsCP": "tscp_coverage_gap_std",
    })
    columns = [
        "dataset", "total_calibration_size",
        "ecp_estimated_coverage", "ecp_empirical_coverage",
        "ecp_coverage_gap", "ecp_coverage_gap_std",
        "tscp_estimated_coverage", "tscp_empirical_coverage",
        "tscp_coverage_gap", "tscp_coverage_gap_std",
    ]
    return table[columns].sort_values(["dataset", "total_calibration_size"]).reset_index(drop=True)


def make_figures(frame: pd.DataFrame, config: dict, output: Path) -> None:
    kind = config["experiment"]["type"]
    datasets = config["datasets"]
    if kind == "self_validation":
        cov = _mean(frame[frame.panel == "coverage"], ["dataset", "x"], ["empirical", "corrected_bound", "old_proxy"])
        size_stats = (
            frame[frame.panel == "size"]
            .groupby(["dataset", "x"], as_index=False)
            .agg(
                average_size=("average_size", "mean"),
                average_size_std=("average_size", "std"),
                budget=("budget", "mean"),
            )
        )
        size_stats["average_size_std"] = size_stats["average_size_std"].fillna(0.0)

        # Coverage figure: one panel per dataset.  The empirical curve varies
        # with the test-batch prefix, whereas the independent-reference target
        # is constant in the number of test samples.
        coverage_fig, coverage_axes = plt.subplots(
            1, len(datasets), figsize=_plot_figsize(config, (5.2 * len(datasets), 4.5)),
            squeeze=False, sharey=True,
        )
        for col, dataset in enumerate(datasets):
            ax = coverage_axes[0, col]
            ax._tscp_plot_scope = "coverage"
            part = cov[cov.dataset == dataset]
            ax.plot(part.x, part.empirical, marker="o", label="Empirical")
            ax.plot(part.x, part.corrected_bound, linestyle="--", label="Theoretical")
            ax.set_title(_dataset_display_name(dataset))
            ax.set_xlabel("Number of test samples")
            if col == 0:
                ax.set_ylabel("Coverage")
                ax.legend(loc="lower left")
            ax.grid(alpha=0.25)
        # "figure" is retained as a backward-compatible alias.
        _save_figure(coverage_fig, output, ("coverage", "figure"), config)

        # Prediction-size figure: again use one panel per dataset so datasets
        # with very different size scales do not share one axis.
        size_fig, size_axes = plt.subplots(
            1, len(datasets), figsize=_plot_figsize(config, (5.2 * len(datasets), 4.5)), squeeze=False,
        )
        for col, dataset in enumerate(datasets):
            ax = size_axes[0, col]
            ax._tscp_plot_scope = "size"
            part = size_stats[size_stats.dataset == dataset].sort_values("x")
            x_values = part.x.to_numpy(dtype=float)
            size_mean = part.average_size.to_numpy(dtype=float)
            size_std = part.average_size_std.to_numpy(dtype=float)
            mean_line, = ax.plot(x_values, size_mean, marker="o", label="Average size")
            ax.fill_between(
                x_values, size_mean - size_std, size_mean + size_std,
                color=mean_line.get_color(), alpha=0.2,
            )
            ax.plot(
                x_values, part.budget.to_numpy(dtype=float), linestyle=":",
                color=mean_line.get_color(), label="Budget",
            )
            ax.set_title(_dataset_display_name(dataset))
            ax.set_xlabel(r"Total calibration size $2n$")
            if col == 0:
                ax.set_ylabel("Average prediction-set size")
                ax.legend(loc="best")
            ax.grid(alpha=0.25)
        _save_figure(size_fig, output, "average_size", config)

        # Also export one publication-ready 1x2 figure per dataset.  The raw
        # dataset key remains in metrics.csv; the display name is used only as
        # the filename.  No panel title is added.
        for dataset in datasets:
            dataset_fig, dataset_axes = plt.subplots(
                1, 2, figsize=_plot_figsize(config, (10.0, 4.0)), squeeze=False,
            )
            coverage_ax, size_ax = dataset_axes[0]
            coverage_ax._tscp_plot_scope = "coverage"
            size_ax._tscp_plot_scope = "size"

            coverage_part = cov[cov.dataset == dataset].sort_values("x")
            coverage_ax.plot(
                coverage_part.x, coverage_part.empirical,
                marker="o", label="Empirical",
            )
            coverage_ax.plot(
                coverage_part.x, coverage_part.corrected_bound,
                linestyle="--", label="Theoretical",
            )
            coverage_ax.set_xlabel("Number of test samples")
            coverage_ax.set_ylabel("Coverage")
            coverage_ax.legend(loc="upper right")
            coverage_ax.grid(alpha=0.25)

            size_part = size_stats[size_stats.dataset == dataset].sort_values("x")
            x_values = size_part.x.to_numpy(dtype=float)
            size_mean = size_part.average_size.to_numpy(dtype=float)
            size_std = size_part.average_size_std.to_numpy(dtype=float)
            mean_line, = size_ax.plot(
                x_values, size_mean, marker="o", label="Mean over seeds",
            )
            size_ax.fill_between(
                x_values, size_mean - size_std, size_mean + size_std,
                color=mean_line.get_color(), alpha=0.2,
            )
            size_ax.plot(
                x_values, size_part.budget.to_numpy(dtype=float),
                linestyle=":", color=mean_line.get_color(),
                label="Pre-chosen set size",
            )
            size_ax.set_xlabel(r"Total calibration size $(2 \times n)$")
            size_ax.set_ylabel("Average set size")
            size_ax.legend(loc="upper right")
            size_ax.grid(alpha=0.25)

            _save_figure(
                dataset_fig, output, _dataset_display_name(dataset), config,
            )
        return
    elif kind == "delta_ablation":
        fig, axes = plt.subplots(1, 2, figsize=_plot_figsize(config, (12, 4.5)))
        avg = _mean(frame, ["dataset", "delta"], ["coverage", "corrected_bound", "average_size", "budget"])
        for dataset in datasets:
            part = avg[avg.dataset == dataset]
            display_name = _dataset_display_name(dataset)
            axes[0].plot(part.delta, part.coverage, marker="o", label=f"{display_name} empirical")
            axes[0].plot(part.delta, part.corrected_bound, linestyle="--", label=f"{display_name} corrected theory")
            axes[1].plot(part.delta, part.average_size, marker="o", label=display_name)
            axes[1].plot(part.delta, part.budget, linestyle=":", color=axes[1].lines[-1].get_color())
        axes[0].set(xlabel=r"Slack $\delta$", ylabel="Coverage")
        axes[1].set(xlabel=r"Slack $\delta$", ylabel="Average prediction-set size")
    elif kind in {"compare_ecp", "budget_ablation"}:
        fig, axes = plt.subplots(
            1, len(datasets), figsize=_plot_figsize(config, (6 * len(datasets), 4.5)), squeeze=False,
        )
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
                ax.set(title=_dataset_display_name(dataset), xlabel="Pre-chosen set size", ylabel="Coverage")
            else:
                ax.set(title=_dataset_display_name(dataset), xlabel="Coverage", ylabel="Average prediction-set size")
    elif kind == "model_ablation":
        fig, axes = plt.subplots(
            2, len(datasets), figsize=_plot_figsize(config, (6 * len(datasets), 8)), squeeze=False,
        )
        avg = _mean(frame, ["dataset", "model", "calibration_size"], ["coverage", "average_size", "corrected_bound"])
        for col, dataset in enumerate(datasets):
            for model, part in avg[avg.dataset == dataset].groupby("model"):
                axes[0, col].plot(part.calibration_size, part.coverage, marker="o", label=model)
                axes[1, col].plot(part.calibration_size, part.average_size, marker="o", label=model)
            axes[0, col].set(title=_dataset_display_name(dataset), xlabel=r"Total calibration size $2n$", ylabel="Coverage")
            axes[1, col].set(xlabel=r"Total calibration size $2n$", ylabel="Average prediction-set size")
    elif kind == "runtime":
        fig, axes = plt.subplots(2, 3, figsize=_plot_figsize(config, (15, 8)), squeeze=False)
        avg = _mean(frame, ["dataset", "method", "number_test"], ["runtime_seconds"])
        for ax, dataset in zip(axes.flat, datasets):
            for method, part in avg[avg.dataset == dataset].groupby("method"):
                ax.plot(part.number_test, part.runtime_seconds, marker="o", label=method)
            ax.set(title=_dataset_display_name(dataset), xlabel="Number of test samples", ylabel="Runtime (seconds)")
        for ax in axes.flat[len(datasets):]:
            ax.axis("off")
    elif kind == "loo_validation":
        fig, axes = plt.subplots(1, 2, figsize=_plot_figsize(config, (12, 4.5)))
        avg = _mean(frame, ["dataset", "calibration_size"], ["alpha_abs_error", "delta_abs_error"])
        for dataset in datasets:
            part = avg[avg.dataset == dataset]
            display_name = _dataset_display_name(dataset)
            axes[0].plot(part.calibration_size, part.alpha_abs_error, marker="o", label=display_name)
            axes[1].plot(part.calibration_size, part.delta_abs_error, marker="o", label=display_name)
        axes[0].set(xlabel=r"Total calibration size $2n$", ylabel=r"$|\hat\alpha-E[\alpha]|$")
        axes[1].set(xlabel=r"Total calibration size $2n$", ylabel=r"$|\hat\Delta-E[\Delta]|$")
    elif kind == "loo_histogram":
        calibration_sizes = [int(value) for value in config["data"]["total_calibration_sizes"]]
        fig, axes = plt.subplots(
            len(datasets), len(calibration_sizes),
            figsize=_plot_figsize(config, (4.2 * len(calibration_sizes), 3.0 * len(datasets))),
            squeeze=False, sharex=True, sharey=True,
        )
        bins = int(config.get("experiment", {}).get("bins", 30))
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
                ax.hist(part.test_corrected, bins=histogram_bins, alpha=0.45,
                        label=r"$1-\hat\alpha_{\rm test}-\hat\delta_{\rm test}$" if show_legend else "_nolegend_",
                        color="tab:blue")
                ax.hist(part.loo_coverage, bins=histogram_bins, alpha=0.45,
                        label=r"$1-\hat\alpha^{\mathrm{LOO}}-\hat\delta^{\mathrm{LOO}}$" if show_legend else "_nolegend_",
                        color="tab:orange")
                ax.axvline(part.test_corrected.mean(), color="tab:blue", linestyle="--", linewidth=1.5)
                ax.axvline(part.loo_coverage.mean(), color="tab:orange", linestyle="--", linewidth=1.5)
                ax.set_xlim(histogram_bins[0], histogram_bins[-1])
                if row == 0:
                    ax.set_title(rf"$N_{{\mathrm{{cal}}}}={size}$")
                if col == 0:
                    ax.set_ylabel(f"{_dataset_display_name(dataset)}\nFrequency")
                if row == len(datasets) - 1 and col == 0:
                    ax.set_xlabel("Coverage")
                if show_legend:
                    ax.legend(fontsize=7)
    elif kind == "loo_compare_ecp":
        fig, axes = plt.subplots(
            3, len(datasets), figsize=_plot_figsize(config, (5.2 * len(datasets), 10.0)), squeeze=False,
        )
        per_seed, summary = summarize_loo_compare(frame)
        per_seed.to_csv(output / "loo_compare_points_by_seed.csv", index=False)
        loo_compare_report_table(summary).to_csv(output / "loo_compare_points.csv", index=False)
        loo_coverage_report_table(summary).to_csv(output / "loo_coverage_points.csv", index=False)
        styles = {
            "truncated_eCP": ("tab:blue", "o", r"truncated eCP: $\hat\alpha^{LOO}$"),
            "TsCP": ("tab:orange", "s", r"TsCP: $\hat\alpha^{LOO}+\hat\delta^{LOO}$"),
        }
        for col, dataset in enumerate(datasets):
            for method_name in ("truncated_eCP", "TsCP"):
                part = summary[(summary.dataset == dataset) & (summary.method == method_name)].sort_values("calibration_size")
                color, marker, label = styles[method_name]
                x = part.calibration_size.to_numpy(dtype=float)
                variance = part.variance_mean.to_numpy(dtype=float)
                variance_std = part.variance_std.to_numpy(dtype=float)
                error = part.absolute_error_mean.to_numpy(dtype=float)
                error_std = part.absolute_error_std.to_numpy(dtype=float)
                coverage_gap = part.coverage_gap_mean.to_numpy(dtype=float)
                coverage_gap_std = part.coverage_gap_std.to_numpy(dtype=float)
                axes[0, col].plot(x, variance, color=color, marker=marker, label=label)
                axes[0, col].fill_between(
                    x, np.maximum(variance - variance_std, np.finfo(float).tiny),
                    variance + variance_std, color=color, alpha=0.18,
                )
                axes[1, col].plot(x, error, color=color, marker=marker, label=label)
                axes[1, col].fill_between(
                    x, np.maximum(error - error_std, 0.0), error + error_std,
                    color=color, alpha=0.18,
                )
                axes[2, col].plot(x, coverage_gap, color=color, marker=marker, label=label)
                axes[2, col].fill_between(
                    x, np.maximum(coverage_gap - coverage_gap_std, 0.0),
                    coverage_gap + coverage_gap_std, color=color, alpha=0.18,
                )
            axes[0, col].set_title(_dataset_display_name(dataset))
            axes[0, col].set_xscale("log")
            axes[0, col].set_yscale("log")
            axes[0, col].set_xlabel(r"Total calibration size $N_{\mathrm{cal}}=2n$")
            axes[1, col].set_xlabel(r"Total calibration size $N_{\mathrm{cal}}=2n$")
            axes[2, col].set_xlabel(r"Total calibration size $N_{\mathrm{cal}}=2n$")
            if col == 0:
                axes[0, col].set_ylabel(r"$\mathrm{Var}(\hat\theta^{LOO})$")
                axes[1, col].set_ylabel(r"$|\hat\theta^{LOO}-\widehat{\mathbb{E}}[\theta]|$")
                axes[2, col].set_ylabel(r"$|\widehat{\mathrm{Coverage}}-\mathrm{Coverage}_{emp}|$")
            if col == 0:
                axes[0, col].legend()
    else:
        return
    for ax in fig.axes:
        if ax.has_data():
            ax.grid(alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=7)
    _save_figure(fig, output, "figure", config)


def write_hard_table(frame: pd.DataFrame, output: Path) -> None:
    table = frame.groupby(["calibration_size", "dataset"])["hard_accuracy"].mean().unstack("dataset")
    table.to_csv(output / "hard_constraint_table.csv")
    (output / "hard_constraint_table.tex").write_text(table.to_latex(float_format="%.3f"), encoding="utf-8")


def environment_metadata() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}
