from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscp.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache experiment datasets.")
    parser.add_argument("datasets", nargs="+", choices=["california_housing", "superconductivity", "covertype", "mnist"])
    args = parser.parse_args()
    for name in args.datasets:
        x, y, task = load_dataset(name)
        print(f"{name}: task={task}, X={x.shape}, y={y.shape}")


if __name__ == "__main__":
    main()

