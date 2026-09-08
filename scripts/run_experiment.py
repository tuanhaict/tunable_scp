from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscp.config import apply_overrides, dump_config, load_config
from tscp.experiments import COLLECTORS, environment_metadata, make_figures, write_hard_table


def main() -> Path:
    parser = argparse.ArgumentParser(description="Run one TsCP experiment from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="Override a dotted YAML key, e.g. method.delta=0.2")
    parser.add_argument("--smoke", action="store_true", help="Use one seed, small synthetic data, calibration and test sizes.")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    if args.smoke:
        task = config.get("task", "regression")
        synthetic = "synthetic_regression" if task == "regression" else "synthetic_classification"
        config["datasets"] = [synthetic]
        config["seeds"] = [0]
        config.setdefault("data", {})["max_samples"] = 1000
        config["data"]["total_calibration_size"] = 100
        config["data"]["total_calibration_sizes"] = [60, 100]
        config["data"]["number_test_samples"] = [20, 50]
        config["data"]["fixed_number_test_samples"] = 50
        config.setdefault("budget", {})["value"] = 10.0 if task == "regression" else 4
        config["budget"].pop("value_by_dataset", None)
        if config.get("experiment", {}).get("type") == "compare_ecp":
            config["experiment"]["budget_values"] = {synthetic: ([6, 10] if task == "regression" else [3, 4])}
        if config.get("experiment", {}).get("type") == "model_ablation":
            config["experiment"]["models"] = ["ridge"] if task == "regression" else ["logistic"]
        if config.get("experiment", {}).get("type") == "loo_histogram":
            config["experiment"]["trials"] = 3
            config["experiment"]["bins"] = 5
        if config.get("experiment", {}).get("type") == "loo_compare_ecp":
            config["experiment"]["trials"] = 3
            config["experiment"]["reference_trials"] = 5
            config["data"]["fixed_number_test_samples"] = 1

    kind = config["experiment"]["type"]
    if kind not in COLLECTORS:
        raise ValueError(f"Unknown experiment type {kind!r}; expected one of {sorted(COLLECTORS)}")
    root = ROOT / config.get("output", {}).get("root", "outputs") / config["experiment"].get("name", kind)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = root / run_id
    output.mkdir(parents=True, exist_ok=False)
    dump_config(config, output / "config.resolved.yaml")
    (output / "environment.json").write_text(json.dumps(environment_metadata(), indent=2), encoding="utf-8")

    frame = COLLECTORS[kind](config)
    frame.to_csv(output / "metrics.csv", index=False)
    if kind == "hard_constraint":
        write_hard_table(frame, output)
    else:
        make_figures(frame, config, output)
    print(output)
    return output


if __name__ == "__main__":
    main()
