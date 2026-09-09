from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscp.config import add_plot_arguments, apply_plot_arguments, load_config
from tscp.experiments import make_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures from saved raw metrics.")
    parser.add_argument("--run-dir", required=True)
    add_plot_arguments(parser)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = apply_plot_arguments(load_config(run_dir / "config.resolved.yaml"), args)
    make_figures(pd.read_csv(run_dir / "metrics.csv"), config, run_dir)


if __name__ == "__main__":
    main()
