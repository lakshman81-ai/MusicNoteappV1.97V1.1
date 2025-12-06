from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

from music21 import converter, note as m21note

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backend.transcription import transcribe_audio_pipeline

NoteTuple = Tuple[int, float, float]  # (midi, onset_beats, duration_beats)


@dataclass
class BenchmarkCase:
    """Configuration for a single audio/reference pair."""

    name: str
    audio_path: Path
    reference_path: Path
    scenario: str

    @property
    def slug(self) -> str:
        cleaned = self.name.lower().replace("–", "-")
        return "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in cleaned.replace(" ", "_")
        ).strip("_")


@dataclass
class MatchMetrics:
    precision: float
    recall: float
    f1: float
    matched: int
    reference_total: int
    predicted_total: int


def extract_notes(score_path: Path) -> List[NoteTuple]:
    score = converter.parse(str(score_path))
    notes: List[NoteTuple] = []
    for n in score.recurse().notes:
        if isinstance(n, m21note.Note):
            onset = float(n.offset)
            duration = float(n.quarterLength)
            notes.append((int(n.pitch.midi), onset, duration))
    notes.sort(key=lambda x: x[1])
    return notes


def match_accuracy(
    reference: List[NoteTuple],
    predicted: List[NoteTuple],
    tol_beats: float = 0.25,
    require_duration: bool = False,
) -> MatchMetrics:
    matched = 0
    used_pred: set[int] = set()

    for midi, onset, duration in reference:
        for idx, (p_midi, p_onset, p_duration) in enumerate(predicted):
            if idx in used_pred:
                continue
            if midi != p_midi:
                continue
            if abs(onset - p_onset) > tol_beats:
                continue
            if require_duration and abs(duration - p_duration) > tol_beats:
                continue
            matched += 1
            used_pred.add(idx)
            break

    reference_total = len(reference)
    predicted_total = len(predicted)
    precision = matched / predicted_total if predicted_total else 0.0
    recall = matched / reference_total if reference_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return MatchMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=matched,
        reference_total=reference_total,
        predicted_total=predicted_total,
    )


def run_single_case(
    case: BenchmarkCase,
    *,
    use_crepe: bool = False,
    output_dir: Path | None = None,
    tempo_override: float | None = None,
    beat_times_override: List[float] | None = None,
) -> Dict[str, float | int | str]:
    """Run transcription for a single audio/reference pair."""

    output_dir = output_dir or Path("benchmarks_results")
    output_dir.mkdir(exist_ok=True)

    if not case.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {case.audio_path}")
    if not case.reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {case.reference_path}")

    print(f"\n🎧 Audio: {case.audio_path}")
    print(f"🎼 Reference: {case.reference_path}")

    result = transcribe_audio_pipeline(
        str(case.audio_path),
        use_crepe=use_crepe,
        tempo_override=tempo_override,
        beat_times_override=beat_times_override,
    )
    musicxml_text = result["musicxml"]
    midi_bytes: bytes = result["midi_bytes"]

    pred_xml_path = output_dir / f"{case.slug}_pred.musicxml"
    pred_mid_path = output_dir / f"{case.slug}_pred.mid"

    pred_xml_path.write_text(musicxml_text, encoding="utf-8")
    pred_mid_path.write_bytes(midi_bytes)

    ref_notes = extract_notes(case.reference_path)
    pred_notes = extract_notes(pred_xml_path)

    pitch_stats = match_accuracy(ref_notes, pred_notes, tol_beats=0.25, require_duration=False)
    rhythm_stats = match_accuracy(ref_notes, pred_notes, tol_beats=0.25, require_duration=True)

    summary: Dict[str, float | int | str] = {
        "name": case.name,
        "scenario": case.scenario,
        "audio": str(case.audio_path),
        "reference": str(case.reference_path),
        "reference_notes": len(ref_notes),
        "predicted_notes": len(pred_notes),
        "pitch_precision": pitch_stats.precision,
        "pitch_recall": pitch_stats.recall,
        "pitch_f1": pitch_stats.f1,
        "rhythm_precision": rhythm_stats.precision,
        "rhythm_recall": rhythm_stats.recall,
        "rhythm_f1": rhythm_stats.f1,
        "pitch_accuracy": pitch_stats.f1,
        "rhythm_accuracy": rhythm_stats.f1,
        "predicted_musicxml": str(pred_xml_path),
        "predicted_midi": str(pred_mid_path),
    }

    summary_path = output_dir / f"{case.slug}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== BENCHMARK RESULT ===")
    print(f"Reference notes: {len(ref_notes)}")
    print(f"Predicted notes: {len(pred_notes)}")
    print(
        "Pitch precision/recall/F1: "
        f"{pitch_stats.precision * 100:.1f}% / {pitch_stats.recall * 100:.1f}% / {pitch_stats.f1 * 100:.1f}%"
    )
    print(
        "Rhythm precision/recall/F1: "
        f"{rhythm_stats.precision * 100:.1f}% / {rhythm_stats.recall * 100:.1f}% / {rhythm_stats.f1 * 100:.1f}% (±0.25 beats)"
    )
    print(f"Predicted MusicXML saved to: {pred_xml_path}")
    print(f"Predicted MIDI saved to: {pred_mid_path}")
    print(f"Per-clip summary saved to: {summary_path}")

    return summary


def run_suite(
    cases: Iterable[BenchmarkCase],
    *,
    use_crepe: bool = False,
    output_dir: Path | None = None,
    threshold: float = 0.75,
    report_path: Path | None = None,
    tempo_override: float | None = None,
    beat_times_override: List[float] | None = None,
) -> int:
    output_dir = output_dir or Path("benchmarks_results")
    output_dir.mkdir(exist_ok=True)

    summaries = [
        run_single_case(
            case,
            use_crepe=use_crepe,
            output_dir=output_dir,
            tempo_override=tempo_override,
            beat_times_override=beat_times_override,
        )
        for case in cases
    ]

    def _avg(key: str) -> float:
        return sum(float(item[key]) for item in summaries) / float(len(summaries))

    avg_pitch_f1 = _avg("pitch_f1")
    avg_rhythm_f1 = _avg("rhythm_f1")

    suite_summary = {
        "average_pitch_accuracy": avg_pitch_f1,
        "average_rhythm_accuracy": avg_rhythm_f1,
        "average_pitch_precision": _avg("pitch_precision"),
        "average_pitch_recall": _avg("pitch_recall"),
        "average_pitch_f1": avg_pitch_f1,
        "average_rhythm_precision": _avg("rhythm_precision"),
        "average_rhythm_recall": _avg("rhythm_recall"),
        "average_rhythm_f1": avg_rhythm_f1,
        "threshold": threshold,
        "cases": summaries,
    }

    aggregate_path = output_dir / "accuracy_suite_summary.json"
    aggregate_path.write_text(json.dumps(suite_summary, indent=2), encoding="utf-8")

    print("\n=== SUITE SUMMARY ===")
    print(
        "Avg pitch precision/recall/F1: "
        f"{suite_summary['average_pitch_precision']*100:.1f}% / "
        f"{suite_summary['average_pitch_recall']*100:.1f}% / "
        f"{avg_pitch_f1*100:.1f}%"
    )
    print(
        "Avg rhythm precision/recall/F1: "
        f"{suite_summary['average_rhythm_precision']*100:.1f}% / "
        f"{suite_summary['average_rhythm_recall']*100:.1f}% / "
        f"{avg_rhythm_f1*100:.1f}% (±0.25 beats)"
    )
    print(f"Aggregate summary saved to: {aggregate_path}")

    if report_path:
        write_markdown_report(suite_summary, report_path)
        print(f"Markdown report saved to: {report_path}")

    if avg_pitch_f1 < threshold or avg_rhythm_f1 < threshold:
        print(
            f"Suite accuracy below threshold ({threshold*100:.0f}%). "
            "Failing with non-zero exit code."
        )
        return 1

    return 0


def write_markdown_report(suite_summary: Dict[str, object], report_path: Path) -> None:
    header = [
        "# Benchmark accuracy snapshot",
        "",
        "Aggregated pitch and rhythm accuracy measured with `benchmarks/benchmark_local_file.py --suite`.",
        "",
        "| Scenario | Audio | Reference | Pitch P / R / F1 | Rhythm P / R / F1 (±0.25 beat) | Notes |",
        "|----------|-------|-----------|------------------|---------------------------------|-------|",
    ]

    rows = []
    for case in suite_summary.get("cases", []):
        pitch_prf = (
            float(case["pitch_precision"]) * 100,
            float(case["pitch_recall"]) * 100,
            float(case["pitch_f1"]) * 100,
        )
        rhythm_prf = (
            float(case["rhythm_precision"]) * 100,
            float(case["rhythm_recall"]) * 100,
            float(case["rhythm_f1"]) * 100,
        )
        notes = (
            f"{case['predicted_notes']} predicted vs {case['reference_notes']} reference notes. "
            f"Files: `{Path(case['predicted_musicxml']).name}` / `{Path(case['predicted_midi']).name}`."
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    str(case["scenario"]),
                    Path(str(case["audio"])).name,
                    Path(str(case["reference"])).as_posix(),
                    f"{pitch_prf[0]:.1f}% / {pitch_prf[1]:.1f}% / {pitch_prf[2]:.1f}%",
                    f"{rhythm_prf[0]:.1f}% / {rhythm_prf[1]:.1f}% / {rhythm_prf[2]:.1f}%",
                    notes,
                ]
            )
            + " |"
        )

    footer = [
        "",
        "## Averages",
        (
            "- Pitch: "
            f"precision {float(suite_summary['average_pitch_precision'])*100:.1f}%, "
            f"recall {float(suite_summary['average_pitch_recall'])*100:.1f}%, "
            f"F1 {float(suite_summary['average_pitch_f1'])*100:.1f}%"
        ),
        (
            "- Rhythm: "
            f"precision {float(suite_summary['average_rhythm_precision'])*100:.1f}%, "
            f"recall {float(suite_summary['average_rhythm_recall'])*100:.1f}%, "
            f"F1 {float(suite_summary['average_rhythm_f1'])*100:.1f}%"
        ),
        f"- Threshold: {float(suite_summary['threshold'])*100:.0f}%",
    ]

    report_path.write_text("\n".join(header + rows + footer) + "\n", encoding="utf-8")


def get_default_suite() -> List[BenchmarkCase]:
    audio_dir = REPO_ROOT / "benchmarks" / "audio"
    ref_dir = REPO_ROOT / "benchmarks" / "reference"
    return [
        BenchmarkCase(
            name="Simple Scale – C Major",
            audio_path=audio_dir / "Simple Scale – C Major.mp3",
            reference_path=ref_dir / "c_major_scale.musicxml",
            scenario="01_scales",
        ),
        BenchmarkCase(
            name="Twinkle Twinkle Little Star",
            audio_path=audio_dir / "Twinkle_Twinkle_Little_Star.mp3",
            reference_path=ref_dir / "twinkle_twinkle.musicxml",
            scenario="02_simple_melodies",
        ),
        BenchmarkCase(
            name="Happy Birthday",
            audio_path=audio_dir / "Happy_birthday_to_you.mp3",
            reference_path=ref_dir / "HappyBirthday.mid",
            scenario="02_simple_melodies",
        ),
        BenchmarkCase(
            name="Ode To Joy",
            audio_path=audio_dir / "Ode To Joy.mp3",
            reference_path=ref_dir / "ode_to_joy.musicxml",
            scenario="03_melody_plus_chords",
        ),
        BenchmarkCase(
            name="Amazing Grace (Familiar Style)",
            audio_path=audio_dir / "AmazingGraceFamiliarStyle.mp3",
            reference_path=ref_dir / "amazing_grace.musicxml",
            scenario="04_pop_loops",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local audio transcription against a reference MusicXML file."
    )
    parser.add_argument("audio_path", nargs="?", type=Path, help="Path to audio file (wav/mp3/etc.)")
    parser.add_argument("reference_path", nargs="?", type=Path, help="Path to reference MusicXML file")
    parser.add_argument("--use-crepe", action="store_true", help="Use CREPE for pitch tracking if available")
    parser.add_argument("--suite", action="store_true", help="Run the default benchmark suite across numbered scenarios")
    parser.add_argument("--threshold", type=float, default=0.75, help="Fail if average pitch or rhythm accuracy falls below this value")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks_results"), help="Where to store predictions and summaries")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks" / "results_accuracy.md", help="Path to write the markdown results table when running the suite")
    parser.add_argument("--tempo-bpm", type=float, default=None, help="Override tempo (BPM) for quantization when beat tracking is unreliable")
    parser.add_argument(
        "--beat-times",
        type=str,
        default=None,
        help="Comma-separated beat times in seconds to override the beat grid (e.g., '0,0.5,1.0,1.5')",
    )
    args = parser.parse_args()

    beat_times_override = None
    if args.beat_times:
        beat_times_override = [float(v) for v in args.beat_times.split(",") if v.strip()]

    if args.suite:
        cases = get_default_suite()
        exit_code = run_suite(
            cases,
            use_crepe=args.use_crepe,
            output_dir=args.output_dir,
            threshold=args.threshold,
            report_path=args.report,
            tempo_override=args.tempo_bpm,
            beat_times_override=beat_times_override,
        )
        raise SystemExit(exit_code)

    if not args.audio_path or not args.reference_path:
        raise SystemExit("Provide audio_path and reference_path or use --suite for batch mode.")

    if not args.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    if not args.reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference_path}")

    case = BenchmarkCase(
        name=args.audio_path.stem,
        audio_path=args.audio_path,
        reference_path=args.reference_path,
        scenario="ad-hoc",
    )
    run_single_case(
        case,
        use_crepe=args.use_crepe,
        output_dir=args.output_dir,
        tempo_override=args.tempo_bpm,
        beat_times_override=beat_times_override,
    )


if __name__ == "__main__":
    main()
