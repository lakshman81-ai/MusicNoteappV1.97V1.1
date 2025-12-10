from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "sample_rate": 44100,
    "fallback_sample_rate": 22050,
    "fmin": 65.0,
    "fmax": 2093.0,
    "yin_threshold": 0.10,
    "hop_length": 512,
    "frame_length": 2048,
    "median_window": 11,
    "ensemble_weights": {
        "swift": 0.45,
        "crepe": 0.25,
        "rmvpe": 0.25,
        "yin": 0.15,
        "cqt": 0.10,
        "autocorr": 0.10,
    },
    "autocorr_split_hz": 1000.0,
    "autocorr_whitening_order": 12,
    "autocorr_whitening_lambda": 0.01,
    "autocorr_peak_threshold": 0.20,
    "autocorr_octave_compression": 2.0,
    "autocorr_octave_suppression": 0.50,
    "onset_threshold_factor": 0.25,
    "min_note_frames": 3,
    "ppq": 120,
    "bar_ticks": 480,
    "min_note_ticks": 30,
    "min_grid_ms": 4.0,
    "target_lufs": -14.0,
    "silence_threshold_dbfs": 40.0,
    "min_duration_sec": 0.12,
    "separation": {
        "enabled": False,
        "model": "htdemucs",
        "device": "auto",
        "normalize": True,
        "target_lufs": -14.0,
    },
    "confidence_floor": 0.10,
    "merge_gap_beats": 0.125,
    "min_staccato_ticks": 30,
    "gross_error_cents": 100.0,
    "pitch_tolerance_cents": 50.0,
    "offset_tolerance_sec": 0.08,
    "onset_tolerance_sec": 0.05,
}

_CONFIG_CACHE: Dict[str, Any] | None = None

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, *, refresh: bool = False) -> Dict[str, Any]:
    """Load the configuration JSON, merged with defaults."""

    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not refresh and path is None:
        return deepcopy(_CONFIG_CACHE)

    config_path = Path(path) if path is not None else CONFIG_PATH
    if not config_path.exists():
        _CONFIG_CACHE = deepcopy(DEFAULT_CONFIG)
        return deepcopy(_CONFIG_CACHE)

    with open(config_path, "r", encoding="utf-8") as f:
        file_config = json.load(f)

    _CONFIG_CACHE = _deep_merge(DEFAULT_CONFIG, file_config)
    return deepcopy(_CONFIG_CACHE)


def save_config(config: Dict[str, Any], path: str | Path | None = None) -> None:
    target_path = Path(path) if path is not None else CONFIG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    # refresh cache
    load_config(path=target_path, refresh=True)


def update_config(updates: Dict[str, Any], path: str | Path | None = None) -> Dict[str, Any]:
    """Apply nested updates to the current config and persist them."""

    current = load_config(path)
    new_config = _deep_merge(current, updates)
    save_config(new_config, path)
    return new_config


def get_config_value(key: str, default: Any | None = None) -> Any:
    """Retrieve a config value using dotted paths (e.g., 'ensemble_weights.swift')."""

    config = load_config()
    parts = key.split(".")
    value: Any = config
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def get_config() -> Dict[str, Any]:
    return load_config()
