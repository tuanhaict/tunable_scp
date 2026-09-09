from pathlib import Path

from tscp.config import load_config
import numpy as np

from tscp.experiments import (
    collect_loo_compare_ecp,
    collect_self_validation,
    loo_compare_report_table,
    loo_coverage_report_table,
    make_figures,
    summarize_loo_compare,
)


def test_self_validation_collector_on_small_synthetic_data(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "experiments" / "self_validation_regression.yaml")
    config["datasets"] = ["synthetic_regression"]
    config["seeds"] = [0]
    config["model"] = {"mean": "ridge", "scale": "ridge"}
    config["experiment"]["reference_trials"] = 4
    config["data"].update({
        "max_samples": 800,
        "total_calibration_size": 80,
        "total_calibration_sizes": [60, 80],
        "number_test_samples": [20, 40],
        "fixed_number_test_samples": 40,
    })
    config["budget"] = {"type": "constant", "value": 10.0}
    frame = collect_self_validation(config)
    assert {"coverage", "size"} == set(frame["panel"])
    coverage = frame[frame["panel"] == "coverage"]
    assert coverage["corrected_bound"].notna().all()
    assert coverage["expected_alpha"].notna().all()
    assert coverage["expected_delta"].notna().all()
    assert set(coverage["reference_trials"]) == {4}
    np.testing.assert_allclose(
        coverage["corrected_bound"],
        1.0 - coverage["expected_alpha"] - coverage["expected_delta"],
    )
    np.testing.assert_allclose(
        coverage["corrected_bound"], coverage["reference_coverage"],
    )
    make_figures(frame, config, tmp_path)
    for name in (
        "coverage.pdf", "coverage.png", "average_size.pdf", "average_size.png",
        "figure.pdf", "figure.png",
    ):
        assert (tmp_path / name).is_file()


def test_loo_comparison_uses_one_random_test_point_per_trial():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "experiments" / "loo_compare_ecp_regression.yaml")
    config["datasets"] = ["synthetic_regression"]
    config["seeds"] = [0]
    config["model"] = {"mean": "ridge", "scale": "ridge"}
    config["experiment"]["trials"] = 2
    config["experiment"]["reference_trials"] = 4
    config["data"].update({"max_samples": 800, "total_calibration_sizes": [60]})
    config["budget"] = {"type": "constant", "value": 10.0}

    frame = collect_loo_compare_ecp(config)

    assert set(frame["number_test"]) == {1}
    assert frame.groupby(["outer_seed", "trial"])["test_index"].nunique().eq(1).all()
    np.testing.assert_allclose(
        frame["absolute_error"], np.abs(frame["loo_estimate"] - frame["reference_target"]),
    )
    np.testing.assert_allclose(
        frame["reference_target"], frame["reference_alpha"] + frame["reference_delta"],
    )
    per_seed, points = summarize_loo_compare(frame)
    assert len(per_seed) == 2
    assert len(points) == 2
    assert {"variance_mean", "absolute_error_mean", "reference_target_mean"} <= set(points.columns)
    report = loo_compare_report_table(points)
    assert list(report.columns) == [
        "dataset", "total_calibration_size", "ecp_variance", "tscp_variance",
        "ecp_absolute_error", "tscp_absolute_error",
    ]
    coverage_report = loo_coverage_report_table(points)
    assert list(coverage_report.columns) == [
        "dataset", "total_calibration_size",
        "ecp_estimated_coverage", "ecp_empirical_coverage",
        "ecp_coverage_gap", "ecp_coverage_gap_std",
        "tscp_estimated_coverage", "tscp_empirical_coverage",
        "tscp_coverage_gap", "tscp_coverage_gap_std",
    ]
