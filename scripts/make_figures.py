from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscp.config import load_config
from tscp.experiments import make_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures from saved raw metrics.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    make_figures(pd.read_csv(run_dir / "metrics.csv"), load_config(run_dir / "config.resolved.yaml"), run_dir)


if __name__ == "__main__":
    main()
