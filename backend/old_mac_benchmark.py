from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backend.config_manager import get_config, update_config
from backend.metrics import compute_metrics, save_metrics
from backend.pipeline.models import NoteEvent
from backend.transcription import transcribe_audio_pipeline
from music21 import converter, tempo


def _load_reference_notes(path: Path) -> List[NoteEvent]:
    """Load reference notes from MusicXML or MIDI using music21 only."""

    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")

    score = converter.parse(str(path))
    tempo_boundaries = score.metronomeMarkBoundaries()
    bpm = None
    if tempo_boundaries:
        _, _, mm = tempo_boundaries[0]
        if isinstance(mm, tempo.MetronomeMark) and mm.number:
            bpm = float(mm.number)
    seconds_per_beat = 60.0 / (bpm if bpm else 120.0)

    notes: List[NoteEvent] = []
    for n in score.recurse().notes:
        onset_beats = float(n.offset)
        duration_beats = float(getattr(n, "quarterLength", 0.0))
        onset = onset_beats * seconds_per_beat
        offset = onset + duration_beats * seconds_per_beat
        notes.append(
            NoteEvent(
                start_sec=onset,
                end_sec=offset,
                midi_note=int(n.pitch.midi),
                pitch_hz=float(n.pitch.frequency),
            )
        )

    notes.sort(key=lambda n: n.start_sec)
    return notes


def _load_predicted_notes(predicted: Iterable[NoteEvent]) -> List[NoteEvent]:
    return sorted(list(predicted), key=lambda n: n.start_sec)


def _note_debug_slice(notes: List[NoteEvent], *, limit: int = 5) -> List[Dict[str, Any]]:
    return [
        {
            "onset_sec": round(n.start_sec, 6),
            "offset_sec": round(n.end_sec, 6),
            "midi_pitch": int(n.midi_note),
            "name": getattr(n, "note_name", None),
        }
        for n in notes[:limit]
    ]


def _maybe_generate_audio(
    audio_path: Path, reference_notes: List[NoteEvent], sample_rate: int
) -> None:
    """
    Create a lightweight synthetic audio file if the expected fixture is missing.

    This keeps the benchmark PR-friendly by avoiding tracked binaries while still
    enabling local runs. A simple sine-wave synthesizer is used to render the
    reference melody with gentle fades to avoid clicks.
    """

    if audio_path.exists():
        return

    if not reference_notes:
        raise ValueError("Reference notes are required to synthesize the audio fixture.")

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    total_duration = max((n.end_sec for n in reference_notes), default=0.0) + 0.1
    num_samples = max(1, int(total_duration * sample_rate))
    buffer = np.zeros(num_samples, dtype=np.float32)

    fade_samples = max(1, int(0.005 * sample_rate))
    for note in reference_notes:
        start_idx = max(0, int(note.start_sec * sample_rate))
        end_idx = max(start_idx + 1, int(note.end_sec * sample_rate))
        if end_idx > buffer.shape[0]:
            pad = end_idx - buffer.shape[0]
            buffer = np.concatenate([buffer, np.zeros(pad, dtype=np.float32)])

        duration = (end_idx - start_idx) / float(sample_rate)
        freq = float(note.pitch_hz or 440.0 * (2 ** ((note.midi_note - 69) / 12)))
        t = np.linspace(0, duration, end_idx - start_idx, endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * freq * t)

        fade = min(fade_samples, tone.size // 2)
        if fade > 0:
            fade_in = np.linspace(0.0, 1.0, fade)
            fade_out = np.linspace(1.0, 0.0, fade)
            tone[:fade] *= fade_in
            tone[-fade:] *= fade_out

        buffer[start_idx:end_idx] += tone.astype(np.float32)

    buffer = np.clip(buffer, -1.0, 1.0)
    int_samples = np.int16(buffer * 32767)

    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


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
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    paths = _prepare_paths(round_number)
    history = _load_history(paths.history_path)
    previous_metrics = history[-1]["metrics"] if history else None

    config = get_config()
    config_snapshot = _config_snapshot(config)
    sample_rate = int(config.get("sample_rate", 44100))

    ref_notes = _load_reference_notes(reference_path)
    _maybe_generate_audio(audio_path, ref_notes, sample_rate)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start_time = time.time()
    result = transcribe_audio_pipeline(str(audio_path))
    elapsed = time.time() - start_time

    pred_notes = _load_predicted_notes(result.get("notes", []))

    metrics = compute_metrics(ref_notes, pred_notes)

    catastrophic_mismatch = all(
        float(metrics.get(k, 0.0)) == 0.0
        for k in ("HM", "RPA", "OnsetF", "VoicingPrecision")
    )
    positive_signal = any(float(metrics.get(k, 0.0)) > 0.0 for k in ("RPA", "CA", "OA", "OnsetF"))

    classification = "catastrophic_mismatch" if catastrophic_mismatch else "tuning_deferred" if not positive_signal else "normal"

    save_metrics(metrics, paths.metrics_path)
    paths.musicxml_path.write_text(result["musicxml"], encoding="utf-8")
    paths.midi_path.write_bytes(result["midi_bytes"])

    history_record = {
        "round": round_number,
        "config": config_snapshot,
        "metrics": metrics,
        "classification": classification,
        "runtime_sec": elapsed,
        "timestamp": time.time(),
    }
    _append_history(paths.history_path, history_record)

    updates: Dict[str, Any] = {}
    debug_info: Dict[str, Any] | None = None

    if catastrophic_mismatch:
        debug_info = {
            "status": "catastrophic_mismatch",
            "reference_notes_preview": _note_debug_slice(ref_notes),
            "predicted_notes_preview": _note_debug_slice(pred_notes),
            "onset_offset_units": "seconds",
            "pitch_units": "MIDI numbers (pitch_hz also available)",
            "actions": [
                "Verify onset/offset units are seconds in both reference and predictions.",
                "Verify pitch comparison uses consistent units (MIDI or Hz).",
            ],
        }
    elif positive_signal:
        updates = _propose_updates(metrics, config, previous_metrics)
    else:
        debug_info = {
            "status": "tuning_deferred",
            "message": "Metric signals are all zero; skipping parameter tuning until a positive score is observed.",
            "reference_notes_preview": _note_debug_slice(ref_notes),
            "predicted_notes_preview": _note_debug_slice(pred_notes),
            "onset_offset_units": "seconds",
            "pitch_units": "MIDI numbers (pitch_hz also available)",
        }

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
        "classification": classification,
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

    if debug_info:
        summary["debug"] = debug_info

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
