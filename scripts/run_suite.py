from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscp.config import add_plot_arguments, forward_plot_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a YAML suite of TsCP experiments.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--smoke", action="store_true")
    add_plot_arguments(parser)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    for relative in suite["experiments"]:
        config = (suite_path.parent / relative).resolve()
        command = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", str(config)]
        if args.smoke:
            command.append("--smoke")
        command.extend(forward_plot_arguments(args))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
