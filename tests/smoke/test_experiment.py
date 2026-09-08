from pathlib import Path

from tscp.config import load_config
import numpy as np

from tscp.experiments import collect_loo_compare_ecp, collect_self_validation


def test_self_validation_collector_on_small_synthetic_data():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "experiments" / "self_validation_regression.yaml")
    config["datasets"] = ["synthetic_regression"]
    config["seeds"] = [0]
    config["model"] = {"mean": "ridge", "scale": "ridge"}
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
    assert frame["corrected_bound"].notna().all()


def test_loo_comparison_uses_one_random_test_point_per_trial():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "experiments" / "loo_compare_ecp_regression.yaml")
    config["datasets"] = ["synthetic_regression"]
    config["seeds"] = [0]
    config["model"] = {"mean": "ridge", "scale": "ridge"}
    config["experiment"]["trials"] = 2
    config["data"].update({"max_samples": 800, "total_calibration_sizes": [60]})
    config["budget"] = {"type": "constant", "value": 10.0}

    frame = collect_loo_compare_ecp(config)

    assert set(frame["number_test"]) == {1}
    assert frame.groupby(["outer_seed", "trial"])["test_index"].nunique().eq(1).all()
    np.testing.assert_allclose(
        frame["absolute_error"], np.abs(frame["loo_estimate"] - frame["test_target"]),
    )
