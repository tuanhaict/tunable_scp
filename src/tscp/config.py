from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


PLOT_ARGUMENTS = (
    ("figsize", "figsize"),
    ("xlim", "xlim"),
    ("ylim", "ylim"),
    ("coverage_xlim", "coverage_xlim"),
    ("coverage_ylim", "coverage_ylim"),
    ("size_xlim", "size_xlim"),
    ("size_ylim", "size_ylim"),
    ("font_size", "font_size"),
    ("title_font_size", "title_font_size"),
    ("label_font_size", "label_font_size"),
    ("tick_font_size", "tick_font_size"),
    ("legend_font_size", "legend_font_size"),
    ("line_width", "line_width"),
    ("marker_size", "marker_size"),
    ("dpi", "dpi"),
    ("max_x_ticks", "max_x_ticks"),
    ("max_y_ticks", "max_y_ticks"),
)

PAIR_PLOT_ARGUMENTS = {
    "figsize", "xlim", "ylim", "coverage_xlim", "coverage_ylim",
    "size_xlim", "size_ylim",
}


def _merge(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parents = config.pop("defaults", [])
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / parent).resolve()
        merged = _merge(merged, load_config(parent_path))
    return _merge(merged, config)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    result = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have key=value form: {item!r}")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        target = result
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return result


def dump_config(config: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def add_plot_arguments(parser) -> None:
    """Add the shared figure-style flags to an argparse parser."""
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--coverage-xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--coverage-ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--size-xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--size-ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--font-size", type=float, help="Base font size; specific font flags override it.")
    parser.add_argument("--title-font-size", type=float)
    parser.add_argument("--label-font-size", type=float)
    parser.add_argument("--tick-font-size", type=float)
    parser.add_argument("--legend-font-size", type=float)
    parser.add_argument("--line-width", type=float)
    parser.add_argument("--marker-size", type=float)
    parser.add_argument("--dpi", type=int, help="DPI for PNG output.")
    parser.add_argument("--max-x-ticks", type=int, help="Maximum intervals on linear x axes.")
    parser.add_argument("--max-y-ticks", type=int, help="Maximum intervals on linear y axes.")


def apply_plot_arguments(config: dict, args) -> dict:
    """Overlay explicitly supplied plotting flags on a loaded config."""
    result = deepcopy(config)
    plot = result.setdefault("plot", {})
    for attribute, key in PLOT_ARGUMENTS:
        value = getattr(args, attribute, None)
        if value is not None:
            plot[key] = list(value) if attribute in PAIR_PLOT_ARGUMENTS else value
    return result


def forward_plot_arguments(args) -> list[str]:
    """Serialize supplied plotting flags for a child run_experiment process."""
    tokens: list[str] = []
    for attribute, _ in PLOT_ARGUMENTS:
        value = getattr(args, attribute, None)
        if value is None:
            continue
        flag = "--" + attribute.replace("_", "-")
        tokens.append(flag)
        if attribute in PAIR_PLOT_ARGUMENTS:
            tokens.extend(str(item) for item in value)
        else:
            tokens.append(str(value))
    return tokens
