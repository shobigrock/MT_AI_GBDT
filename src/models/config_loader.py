"""Utility helpers for loading model hyperparameters from YAML/JSON configs.

This keeps experiment hyperparameters out of the model code and allows
per-mode overrides (e.g., different weighting strategies for MTGBDT).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - handled by runtime guard
    yaml = None


def _find_config_path(model_name: str, config_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    """Return the first matching config path for the given model name."""
    base_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs" / "model"
    candidates = [
        base_dir / f"{model_name}.yaml",
        base_dir / f"{model_name}.yml",
        base_dir / f"{model_name}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_raw_config(path: Path) -> Dict[str, Any]:
    """Load a raw config dict from YAML or JSON."""
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs. Please install pyyaml.")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_config(
    model_name: str,
    mode: Optional[str] = None,
    config_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load and merge hyperparameters for a model.

    - Looks for configs/model/<model_name>.yaml|yml|json relative to project root
      (or ``config_dir`` if provided).
    - Merges ``common`` and the selected mode from ``modes``.
    - If ``mode`` is None, falls back to ``default_mode`` in the file.
    """
    config_path = _find_config_path(model_name, config_dir)
    if config_path is None:
        return {}

    raw_cfg = _load_raw_config(config_path)
    common = raw_cfg.get("common", {}) or {}
    selected_mode = mode or raw_cfg.get("default_mode")
    mode_params: Dict[str, Any] = {}
    if selected_mode:
        mode_params = raw_cfg.get("modes", {}).get(selected_mode, {}) or {}

    merged = {**common, **mode_params}
    return merged
