from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backend.metrics import compute_metrics
from backend.pipeline.models import NoteEvent
from backend.transcription import transcribe_audio_pipeline
from benchmarks.benchmark_local_file import BenchmarkCase, ensure_noisy_audio, load_reference_notes, load_predicted_notes

RESULTS_ROOT = Path("benchmarks_results") / "backend_phases"
AUDIO_DIR = REPO_ROOT / "benchmarks" / "audio"
REFERENCE_DIR = REPO_ROOT / "benchmarks" / "reference"


@dataclass
class PreparedCase:
    name: str
    scenario: str
    audio_path: Path
    reference_notes: List[NoteEvent]
    slug: str


@dataclass
class IterationResult:
    strategy: str
    metrics: Dict[str, float]
    runtime_sec: float
    precision: float
    recall: float
    rpa: float
    rca: float
    details: Dict[str, object]


def _note_events_from_midi(midi_notes: Sequence[int], *, start: float = 0.0, duration: float = 1.0) -> List[NoteEvent]:
    notes: List[NoteEvent] = []
    cursor = float(start)
    for midi_note in midi_notes:
        end = cursor + float(duration)
        notes.append(
            NoteEvent(
                start_sec=cursor,
                end_sec=end,
                midi_note=int(midi_note),
                pitch_hz=float(librosa.midi_to_hz(int(midi_note))),
            )
        )
        cursor = end
    return notes


def synthesize_chord_audio(midi_notes: Sequence[int], *, duration: float = 2.5, sr: int = 44100) -> np.ndarray:
    if not midi_notes:
        return np.zeros(int(sr * duration), dtype=np.float32)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveforms = [np.sin(2 * np.pi * librosa.midi_to_hz(int(note)) * t) for note in midi_notes]
    mix = np.sum(waveforms, axis=0) / float(len(waveforms))
    return mix.astype(np.float32)


def write_audio(path: Path, audio: np.ndarray, sr: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)
    return path


def prepare_case_from_assets(name: str, audio_filename: str, reference_filename: str, scenario: str) -> PreparedCase:
    audio_path = AUDIO_DIR / audio_filename
    reference_path = REFERENCE_DIR / reference_filename
    reference_notes = load_reference_notes(reference_path)
    slug = BenchmarkCase(name=name, audio_path=audio_path, reference_path=reference_path, scenario=scenario).slug
    return PreparedCase(
        name=name,
        scenario=scenario,
        audio_path=audio_path,
        reference_notes=reference_notes,
        slug=slug,
    )


def prepare_noisy_case(base: PreparedCase, *, snr_db: float) -> PreparedCase:
    noisy_audio = ensure_noisy_audio(base.audio_path, float(snr_db), RESULTS_ROOT / "noisy_inputs")
    return PreparedCase(
        name=f"{base.name} (SNR {snr_db}dB)",
        scenario=f"{base.scenario}/SNR{int(snr_db)}",
        audio_path=noisy_audio,
        reference_notes=base.reference_notes,
        slug=f"{base.slug}_snr{int(snr_db)}",
    )


def prepare_polyphonic_case(midi_notes: Sequence[int], *, duration: float, scenario: str) -> PreparedCase:
    audio = synthesize_chord_audio(midi_notes, duration=duration)
    output_dir = RESULTS_ROOT / "generated"
    audio_path = output_dir / f"polyphonic_{'_'.join(str(m) for m in midi_notes)}.wav"
    write_audio(audio_path, audio, 44100)
    reference_notes = _note_events_from_midi(midi_notes, duration=duration / len(midi_notes))
    return PreparedCase(
        name=f"Synthetic polyphony ({len(midi_notes)} voices)",
        scenario=scenario,
        audio_path=audio_path,
        reference_notes=reference_notes,
        slug=f"poly_{len(midi_notes)}voices",
    )


def _aggregate_metrics(results: Iterable[IterationResult]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    collected = list(results)
    if not collected:
        return metrics

    keys = set(collected[0].metrics.keys())
    for item in collected[1:]:
        keys &= set(item.metrics.keys())

    for key in sorted(keys):
        metrics[key] = float(sum(result.metrics.get(key, 0.0) for result in collected)) / float(len(collected))

    metrics.update(
        {
            "Precision": float(sum(r.precision for r in collected)) / float(len(collected)),
            "Recall": float(sum(r.recall for r in collected)) / float(len(collected)),
            "RPA": float(sum(r.rpa for r in collected)) / float(len(collected)),
            "RCA": float(sum(r.rca for r in collected)) / float(len(collected)),
        }
    )
    return metrics


def evaluate_case(case: PreparedCase, *, strategy: str, output_dir: Path, pipeline_kwargs: Dict[str, object]) -> IterationResult:
    start_time = time.perf_counter()
    result = transcribe_audio_pipeline(
        str(case.audio_path),
        use_crepe=bool(pipeline_kwargs.get("use_crepe", False)),
        tempo_override=pipeline_kwargs.get("tempo_override"),
        beat_times_override=pipeline_kwargs.get("beat_times_override"),
    )
    runtime = time.perf_counter() - start_time

    pred_notes = load_predicted_notes(result.get("notes", []))
    metrics = compute_metrics(case.reference_notes, pred_notes)

    precision = float(metrics.get("VoicingPrecision", 0.0))
    recall = float(metrics.get("VoicingRecall", 0.0))
    rpa = float(metrics.get("RPA", 0.0))
    rca = float(metrics.get("CA", metrics.get("OA", 0.0)))
    metrics.update({"Precision": precision, "Recall": recall, "RCA": rca})

    case_dir = output_dir / case.scenario
    case_dir.mkdir(parents=True, exist_ok=True)
    musicxml_path = case_dir / f"{case.slug}_pred.musicxml"
    midi_path = case_dir / f"{case.slug}_pred.mid"
    musicxml_path.write_text(result["musicxml"], encoding="utf-8")
    midi_path.write_bytes(result["midi_bytes"])

    details = {
        "case": case.name,
        "scenario": case.scenario,
        "audio": str(case.audio_path),
        "musicxml_path": str(musicxml_path),
        "midi_path": str(midi_path),
        "reference_notes": len(case.reference_notes),
        "predicted_notes": len(pred_notes),
    }

    return IterationResult(
        strategy=strategy,
        metrics=metrics,
        runtime_sec=runtime,
        precision=precision,
        recall=recall,
        rpa=rpa,
        rca=rca,
        details=details,
    )


def run_iteration(
    cases: Iterable[PreparedCase], *, strategy_name: str, output_dir: Path, pipeline_kwargs: Dict[str, object]
) -> Dict[str, object]:
    per_case: List[IterationResult] = []
    for case in cases:
        per_case.append(
            evaluate_case(
                case,
                strategy=strategy_name,
                output_dir=output_dir,
                pipeline_kwargs=pipeline_kwargs,
            )
        )

    aggregate = _aggregate_metrics(per_case)
    hm = aggregate.get("HM", 0.0)
    summary = {
        "strategy": strategy_name,
        "average_metrics": aggregate,
        "average_hm": hm,
        "cases": [result.details | {"metrics": result.metrics, "runtime_sec": result.runtime_sec} for result in per_case],
    }
    return summary


def save_phase_summary(phase_name: str, summary: Dict[str, object]) -> Path:
    phase_dir = RESULTS_ROOT / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    target = phase_dir / "summary.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return target
