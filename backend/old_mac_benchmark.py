from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backend.config_manager import get_config, update_config
from backend.metrics import compute_metrics, save_metrics
from backend.transcription import transcribe_audio_pipeline
from benchmarks.benchmark_local_file import load_predicted_notes, load_reference_notes


@dataclass
class BenchmarkPaths:
    """Container for frequently used benchmark paths."""

    base_dir: Path
    metrics_path: Path
    history_path: Path
    summary_path: Path
    musicxml_path: Path
    midi_path: Path


def _clamp(value: float, *, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _load_history(history_path: Path) -> List[Dict[str, Any]]:
    if not history_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _append_history(history_path: Path, record: Dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _config_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the parameters most relevant to benchmarking tweaks."""

    ensemble = config.get("ensemble_weights", {})
    return {
        "sample_rate": config.get("sample_rate"),
        "fmin": config.get("fmin"),
        "fmax": config.get("fmax"),
        "hop_length": config.get("hop_length"),
        "frame_length": config.get("frame_length"),
        "median_window": config.get("median_window"),
        "confidence_floor": config.get("confidence_floor"),
        "onset_threshold_factor": config.get("onset_threshold_factor"),
        "min_note_frames": config.get("min_note_frames"),
        "ppq": config.get("ppq"),
        "ensemble_weights": {
            "swift": ensemble.get("swift"),
            "yin": ensemble.get("yin"),
            "crepe": ensemble.get("crepe"),
            "rmvpe": ensemble.get("rmvpe"),
            "cqt": ensemble.get("cqt"),
            "autocorr": ensemble.get("autocorr"),
            "spectral": ensemble.get("spectral"),
        },
    }


def _propose_updates(
    metrics: Dict[str, float],
    config: Dict[str, Any],
    previous_metrics: Dict[str, Any] | None,
) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}

    hm = float(metrics.get("HM", 0.0))
    onset_f = float(metrics.get("OnsetF", 0.0))
    ca = float(metrics.get("CA", 0.0))
    oa = float(metrics.get("OA", 0.0))
    prev_hm = float(previous_metrics.get("HM", -1.0)) if previous_metrics else None

    ensemble = config.get("ensemble_weights", {})
    swift_weight = float(ensemble.get("swift", 0.0))
    yin_weight = float(ensemble.get("yin", 0.0))

    if hm < 0.90 or (prev_hm is not None and hm < prev_hm):
        updates.setdefault("ensemble_weights", {})["swift"] = round(_clamp(swift_weight + 0.05, min_value=0.0, max_value=0.70), 3)
        updates.setdefault("ensemble_weights", {})["yin"] = round(_clamp(yin_weight - 0.05, min_value=0.05, max_value=1.0), 3)
        updates["confidence_floor"] = round(_clamp(float(config.get("confidence_floor", 0.1)) + 0.02, min_value=0.0, max_value=0.25), 3)
        updates["median_window"] = int(min(int(config.get("median_window", 11)) + 2, 21))

    if onset_f < 0.90:
        updates["onset_threshold_factor"] = round(
            _clamp(float(config.get("onset_threshold_factor", 0.25)) + 0.05, min_value=0.0, max_value=0.50),
            3,
        )
        updates["min_note_frames"] = int(min(int(config.get("min_note_frames", 3)) + 1, 6))

    if ca < 0.90 or oa < 0.90:
        current_min_frames = updates.get("min_note_frames", int(config.get("min_note_frames", 3)))
        updates["min_note_frames"] = int(min(current_min_frames + 1, 6))
        updates["fmax"] = round(float(config.get("fmax", 2093.0)) * 0.9, 2)
        updates["fmin"] = max(float(config.get("fmin", 65.0)), 80.0)

    return updates


def _prepare_paths(round_number: int) -> BenchmarkPaths:
    base_dir = Path("benchmarks_results") / "old_macdonald"
    base_dir.mkdir(parents=True, exist_ok=True)
    return BenchmarkPaths(
        base_dir=base_dir,
        metrics_path=base_dir / f"old_macdonald_metrics_round{round_number}.json",
        history_path=base_dir / "history.jsonl",
        summary_path=base_dir / "summary.json",
        musicxml_path=base_dir / f"old_macdonald_round{round_number}_pred.musicxml",
        midi_path=base_dir / f"old_macdonald_round{round_number}_pred.mid",
    )


def run_round(audio_path: Path, reference_path: Path, round_number: int) -> Dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    paths = _prepare_paths(round_number)
    history = _load_history(paths.history_path)
    previous_metrics = history[-1]["metrics"] if history else None

    config = get_config()
    config_snapshot = _config_snapshot(config)

    start_time = time.time()
    result = transcribe_audio_pipeline(str(audio_path))
    elapsed = time.time() - start_time

    ref_notes = load_reference_notes(reference_path)
    pred_notes = load_predicted_notes(result.get("notes", []))

    metrics = compute_metrics(ref_notes, pred_notes)

    save_metrics(metrics, paths.metrics_path)
    paths.musicxml_path.write_text(result["musicxml"], encoding="utf-8")
    paths.midi_path.write_bytes(result["midi_bytes"])

    history_record = {
        "round": round_number,
        "config": config_snapshot,
        "metrics": metrics,
        "runtime_sec": elapsed,
        "timestamp": time.time(),
    }
    _append_history(paths.history_path, history_record)

    updates = _propose_updates(metrics, config, previous_metrics)
    new_config = update_config(updates) if updates else config

    best_hm_round = round_number
    best_hm_value = metrics.get("HM", 0.0)
    best_metrics_path = str(paths.metrics_path)
    for entry in history:
        hm_value = float(entry.get("metrics", {}).get("HM", 0.0))
        if hm_value > best_hm_value:
            best_hm_value = hm_value
            best_hm_round = int(entry.get("round", best_hm_round))
            best_metrics_path = str(paths.base_dir / f"old_macdonald_metrics_round{best_hm_round}.json")

    summary = {
        "latest_round": round_number,
        "latest_metrics_path": str(paths.metrics_path),
        "latest_config": config_snapshot,
        "metrics": metrics,
        "runtime_sec": elapsed,
        "updates_applied": updates,
        "config_after_updates": _config_snapshot(new_config),
        "best_hm": {
            "round": best_hm_round,
            "HM": best_hm_value,
            "metrics_path": best_metrics_path,
        },
        "history_path": str(paths.history_path),
        "musicxml_path": str(paths.musicxml_path),
        "midi_path": str(paths.midi_path),
    }

    paths.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Old MacDonald transcription and refine config")
    parser.add_argument("audio_path", help="Path to the Old MacDonald audio file")
    parser.add_argument("reference_path", help="Path to the ground-truth note file (MIDI/MusicXML/JSON)")
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Round number; defaults to the next number after existing history",
    )
    args = parser.parse_args()

    round_number = args.round
    if round_number is None:
        history_path = (Path("benchmarks_results") / "old_macdonald" / "history.jsonl")
        existing_history = _load_history(history_path)
        round_number = len(existing_history) + 1

    summary = run_round(Path(args.audio_path), Path(args.reference_path), round_number)
    print("\n=== Old MacDonald Benchmark Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
