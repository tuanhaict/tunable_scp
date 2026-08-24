from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


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

