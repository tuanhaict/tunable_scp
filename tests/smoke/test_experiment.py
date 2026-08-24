from pathlib import Path

from tscp.config import load_config
from tscp.experiments import collect_self_validation


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
