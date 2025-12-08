from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

import librosa
from music21 import converter, note as m21note

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backend.transcription import transcribe_audio_pipeline

NoteTuple = Tuple[int, float, float]  # (midi, onset_beats, duration_beats)


@dataclass
class MatchStats:
    accuracy: float
    matched: int
    missed: int
    extra: int
    missed_onsets_sample: List[float]
    extra_onsets_sample: List[float]


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


@dataclass
class SuiteRunResult:
    """Captures the outcome of a suite run."""

    exit_code: int
    summary: Dict[str, object]
    strategy_label: str | None = None


@dataclass
class AgentStrategy:
    """Defines a strategy to try while iterating toward the accuracy target."""

    name: str
    use_crepe: bool
    tempo_override: float | None = None
    beat_times_override: List[float] | None = None

    def describe(self) -> str:
        parts = [self.name]
        parts.append("crepe" if self.use_crepe else "pyin/yin")
        if self.tempo_override is not None:
            parts.append(f"{self.tempo_override:.0f} bpm")
        if self.beat_times_override:
            parts.append(f"{len(self.beat_times_override)} beat anchors")
        return " | ".join(parts)


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
    sample_limit: int = 5,
) -> MatchStats:
    if not reference:
        extra_onsets = [p_onset for _, p_onset, _ in predicted][:sample_limit]
        return MatchStats(
            accuracy=0.0,
            matched=0,
            missed=0,
            extra=len(predicted),
            missed_onsets_sample=[],
            extra_onsets_sample=extra_onsets,
        )

    matched = 0
    used_pred: set[int] = set()
    missed_onsets: List[float] = []

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
        else:
            if len(missed_onsets) < sample_limit:
                missed_onsets.append(onset)

    extra_onsets = [p_onset for idx, (_, p_onset, _) in enumerate(predicted) if idx not in used_pred]

    return MatchStats(
        accuracy=matched / len(reference),
        matched=matched,
        missed=len(reference) - matched,
        extra=len(extra_onsets),
        missed_onsets_sample=missed_onsets,
        extra_onsets_sample=extra_onsets[:sample_limit],
    )


def get_audio_metadata(path: Path) -> tuple[float, int, int]:
    """Return (duration_seconds, sample_rate, channel_count) for an audio file."""

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Load a short snippet (or header info) to infer sample rate and channels without
    # decoding the full file.
    y, sr = librosa.load(str(path), sr=None, mono=False, duration=1.0)
    channels = int(y.shape[0]) if y.ndim > 1 else 1
    duration_seconds = float(librosa.get_duration(filename=str(path)))
    return duration_seconds, int(sr), channels


def format_audio_note(sample_rate: int, channels: int, duration_seconds: float) -> str:
    """Render a concise metadata snippet like '44.1kHz stereo, 18s'."""

    sr_khz = sample_rate / 1000
    channel_label = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels}-channel"
    duration_label = f"{round(duration_seconds)}s"
    return f"{sr_khz:.1f}kHz {channel_label}, {duration_label}"


def calculate_prf(stats: MatchStats) -> MatchMetrics:
    predicted_total = stats.matched + stats.extra
    reference_total = stats.matched + stats.missed

    precision = stats.matched / predicted_total if predicted_total else 0.0
    recall = stats.matched / reference_total if reference_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return MatchMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=stats.matched,
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

    duration_seconds, sample_rate, channels = get_audio_metadata(case.audio_path)

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
    pitch_metrics = calculate_prf(pitch_stats)
    rhythm_metrics = calculate_prf(rhythm_stats)

    summary: Dict[str, float | int | str] = {
        "name": case.name,
        "scenario": case.scenario,
        "audio": str(case.audio_path),
        "reference": str(case.reference_path),
        "audio_duration_seconds": duration_seconds,
        "audio_sample_rate": sample_rate,
        "audio_channels": channels,
        "audio_note": format_audio_note(sample_rate, channels, duration_seconds),
        "reference_notes": len(ref_notes),
        "predicted_notes": len(pred_notes),
        "pitch_accuracy": pitch_stats.accuracy,
        "pitch_precision": pitch_metrics.precision,
        "pitch_recall": pitch_metrics.recall,
        "pitch_f1": pitch_metrics.f1,
        "pitch_matched": pitch_stats.matched,
        "pitch_missed": pitch_stats.missed,
        "pitch_extra": pitch_stats.extra,
        "pitch_missed_onsets_sample": pitch_stats.missed_onsets_sample,
        "pitch_extra_onsets_sample": pitch_stats.extra_onsets_sample,
        "rhythm_accuracy": rhythm_stats.accuracy,
        "rhythm_precision": rhythm_metrics.precision,
        "rhythm_recall": rhythm_metrics.recall,
        "rhythm_f1": rhythm_metrics.f1,
        "rhythm_matched": rhythm_stats.matched,
        "rhythm_missed": rhythm_stats.missed,
        "rhythm_extra": rhythm_stats.extra,
        "rhythm_missed_onsets_sample": rhythm_stats.missed_onsets_sample,
        "rhythm_extra_onsets_sample": rhythm_stats.extra_onsets_sample,
        "predicted_musicxml": str(pred_xml_path),
        "predicted_midi": str(pred_mid_path),
    }

    summary_path = output_dir / f"{case.slug}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== BENCHMARK RESULT ===")
    print(f"Reference notes: {len(ref_notes)}")
    print(f"Predicted notes: {len(pred_notes)}")
    print(
        "Pitch accuracy:  "
        f"{pitch_stats.accuracy * 100:.1f}% | matched {pitch_stats.matched}, "
        f"missed {pitch_stats.missed}, extra {pitch_stats.extra}"
    )
    print(
        "Rhythm accuracy: "
        f"{rhythm_stats.accuracy * 100:.1f}% (±0.25 beats) | "
        f"matched {rhythm_stats.matched}, missed {rhythm_stats.missed}, extra {rhythm_stats.extra}"
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
    strategy_label: str | None = None,
) -> SuiteRunResult:
    output_dir = output_dir or Path("benchmarks_results")
    output_dir.mkdir(exist_ok=True)

    if strategy_label:
        print(f"Applying strategy: {strategy_label}")

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
        "strategy_label": strategy_label or "default",
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

    exit_code = 0
    if avg_pitch_f1 < threshold or avg_rhythm_f1 < threshold:
        print(
            f"Suite accuracy below threshold ({threshold*100:.0f}%). "
            "Failing with non-zero exit code."
        )
        exit_code = 1

    return SuiteRunResult(exit_code=exit_code, summary=suite_summary, strategy_label=strategy_label)


def write_markdown_report(suite_summary: Dict[str, object], report_path: Path) -> None:
    def _prf_from_counts(matched: int, missed: int, extra: int) -> tuple[float, float, float]:
        precision_den = matched + extra
        recall_den = matched + missed

        precision = matched / precision_den if precision_den else 0.0
        recall = matched / recall_den if recall_den else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return precision * 100, recall * 100, f1 * 100

    header = [
        "# Benchmark accuracy snapshot",
        "",
        "Aggregated pitch and rhythm accuracy measured with `benchmarks/benchmark_local_file.py --suite`.",
        "",
        "| Scenario | Audio | Reference | Pitch P / R / F1 | Rhythm P / R / F1 (±0.25 beat) | Notes |",
        "|----------|-------|-----------|------------------|---------------------------------|-------|",
    ]

    rows = []
    threshold = float(suite_summary.get("threshold", 0.75))
    for case in suite_summary.get("cases", []):
        pitch_missed = case.get("pitch_missed", 0)
        pitch_extra = case.get("pitch_extra", 0)
        rhythm_missed = case.get("rhythm_missed", 0)
        rhythm_extra = case.get("rhythm_extra", 0)

        pitch_prf = (
            float(case.get("pitch_precision", 0.0)) * 100,
            float(case.get("pitch_recall", 0.0)) * 100,
            float(case.get("pitch_f1", 0.0)) * 100,
        )
        rhythm_prf = (
            float(case.get("rhythm_precision", 0.0)) * 100,
            float(case.get("rhythm_recall", 0.0)) * 100,
            float(case.get("rhythm_f1", 0.0)) * 100,
        )

        pitch_samples = []
        if case.get("pitch_missed_onsets_sample"):
            pitch_samples.append(
                "missed onsets: " + ", ".join(f"{v:.2f}" for v in case["pitch_missed_onsets_sample"])
            )
        if case.get("pitch_extra_onsets_sample"):
            pitch_samples.append(
                "extra onsets: " + ", ".join(f"{v:.2f}" for v in case["pitch_extra_onsets_sample"])
            )

        rhythm_samples = []
        if case.get("rhythm_missed_onsets_sample"):
            rhythm_samples.append(
                "missed onsets: " + ", ".join(f"{v:.2f}" for v in case["rhythm_missed_onsets_sample"])
            )
        if case.get("rhythm_extra_onsets_sample"):
            rhythm_samples.append(
                "extra onsets: " + ", ".join(f"{v:.2f}" for v in case["rhythm_extra_onsets_sample"])
            )

        pitch_note = f"Pitch matched {case.get('pitch_matched', 0)} (missed {pitch_missed}, extra {pitch_extra})"
        rhythm_note = f"Rhythm matched {case.get('rhythm_matched', 0)} (missed {rhythm_missed}, extra {rhythm_extra})"

        samples_note = []
        if pitch_samples:
            samples_note.append("Pitch " + "; ".join(pitch_samples))
        if rhythm_samples:
            samples_note.append("Rhythm " + "; ".join(rhythm_samples))

        notes_parts = [
            str(case.get("audio_note", "")),
            f"{case['predicted_notes']} predicted vs {case['reference_notes']} reference notes.",
            pitch_note + ".",
            rhythm_note + ".",
            " ".join(samples_note),
            f"Files: `{Path(case['predicted_musicxml']).name}` / `{Path(case['predicted_midi']).name}`.",
        ]

        notes = " ".join(part for part in notes_parts if part).strip()
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


def _parse_tempos(raw: str | None) -> List[float | None]:
    if not raw:
        return [None, 92.0, 100.0, 120.0]

    parsed: List[float | None] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if stripped.lower() == "auto":
            parsed.append(None)
            continue
        try:
            parsed.append(float(stripped))
        except ValueError as exc:  # pragma: no cover - argparse guards input
            raise ValueError(f"Invalid tempo value: {stripped}") from exc
    return parsed or [None]


def build_agent_strategies(
    *,
    base_use_crepe: bool,
    tempos: List[float | None],
    prioritize_crepe: bool,
) -> List[AgentStrategy]:
    seen: set[tuple[bool, float | None]] = set()
    strategies: List[AgentStrategy] = []

    def _add(name: str, use_crepe: bool, tempo_override: float | None) -> None:
        key = (use_crepe, tempo_override)
        if key in seen:
            return
        seen.add(key)
        strategies.append(
            AgentStrategy(
                name=name,
                use_crepe=use_crepe,
                tempo_override=tempo_override,
                beat_times_override=None,
            )
        )

    tempo_labels = {
        None: "auto tempo",
    }

    ordered_tempos = tempos or [None]
    # Optionally front-load CREPE strategies to chase accuracy sooner.
    if prioritize_crepe:
        for tempo in ordered_tempos:
            _add(f"crepe @ {tempo_labels.get(tempo, f'{tempo:.0f} bpm')}", True, tempo)

    for tempo in ordered_tempos:
        _add(f"baseline @ {tempo_labels.get(tempo, f'{tempo:.0f} bpm')}", base_use_crepe, tempo)

    # If CREPE was not prioritized but differs from baseline, try it afterward.
    if not prioritize_crepe and not base_use_crepe:
        for tempo in ordered_tempos:
            _add(f"crepe @ {tempo_labels.get(tempo, f'{tempo:.0f} bpm')}", True, tempo)

    return strategies


def run_agent_mode(
    cases: Iterable[BenchmarkCase],
    *,
    output_dir: Path,
    target_accuracy: float,
    max_runs: int,
    tempos: List[float | None],
    base_use_crepe: bool,
    prioritize_crepe: bool,
    report_path: Path | None,
    beat_times_override: List[float] | None,
) -> int:
    strategies = build_agent_strategies(
        base_use_crepe=base_use_crepe,
        tempos=tempos,
        prioritize_crepe=prioritize_crepe,
    )

    if max_runs > 0:
        strategies = strategies[:max_runs]

    print(
        f"Starting agent mode with {len(strategies)} strategy candidates "
        f"(target F1 ≥ {target_accuracy*100:.1f}%)."
    )

    history: List[Dict[str, object]] = []
    reached_target = False

    for idx, strategy in enumerate(strategies, start=1):
        print(f"\n🤖 Agent iteration {idx}/{len(strategies)}: {strategy.describe()}")
        iter_report_path = None
        if report_path:
            iter_report_path = report_path.with_name(
                f"{report_path.stem}_agent_{idx}{report_path.suffix}"
            )

        suite_result = run_suite(
            cases,
            use_crepe=strategy.use_crepe,
            output_dir=output_dir,
            threshold=target_accuracy,
            report_path=iter_report_path,
            tempo_override=strategy.tempo_override,
            beat_times_override=beat_times_override,
            strategy_label=strategy.name,
        )

        summary = suite_result.summary
        history.append(
            {
                "iteration": idx,
                "strategy": strategy.describe(),
                "average_pitch_f1": summary.get("average_pitch_f1"),
                "average_rhythm_f1": summary.get("average_rhythm_f1"),
                "pitch_precision": summary.get("average_pitch_precision"),
                "pitch_recall": summary.get("average_pitch_recall"),
                "rhythm_precision": summary.get("average_rhythm_precision"),
                "rhythm_recall": summary.get("average_rhythm_recall"),
                "report_path": iter_report_path.as_posix() if iter_report_path else None,
                "exit_code": suite_result.exit_code,
            }
        )

        pitch_ok = float(summary.get("average_pitch_f1", 0.0)) >= target_accuracy
        rhythm_ok = float(summary.get("average_rhythm_f1", 0.0)) >= target_accuracy
        if pitch_ok and rhythm_ok:
            reached_target = True
            print("Target accuracy reached; stopping agent loop.")
            break

    history_path = output_dir / "agent_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Agent history saved to: {history_path}")

    if not reached_target:
        print(
            "Agent mode exhausted the configured strategies without hitting the target accuracy. "
            "Consider adding more tempos or enabling CREPE."
        )

    return 0 if reached_target else 1


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
    parser.add_argument("--agent-mode", action="store_true", help="Iterate through strategies from simple to complex until the target accuracy is reached")
    parser.add_argument(
        "--agent-target",
        type=float,
        default=0.85,
        help="Target average pitch/rhythm F1 to stop when running with --agent-mode",
    )
    parser.add_argument(
        "--agent-max-runs",
        type=int,
        default=5,
        help="Maximum number of strategy iterations to attempt in --agent-mode",
    )
    parser.add_argument(
        "--agent-tempos",
        type=str,
        default=None,
        help="Comma-separated tempos to try in --agent-mode (include 'auto' to keep beat tracking)",
    )
    parser.add_argument(
        "--agent-prioritize-crepe",
        action="store_true",
        help="When set, try CREPE-powered strategies before the baseline in --agent-mode",
    )
    args = parser.parse_args()

    beat_times_override = None
    if args.beat_times:
        beat_times_override = [float(v) for v in args.beat_times.split(",") if v.strip()]

    if args.agent_mode and not args.suite:
        raise SystemExit("--agent-mode requires --suite so it can iterate across the benchmark scenarios.")

    if args.suite:
        cases = get_default_suite()
        if args.agent_mode:
            agent_tempos = _parse_tempos(args.agent_tempos)
            exit_code = run_agent_mode(
                cases,
                output_dir=args.output_dir,
                target_accuracy=args.agent_target,
                max_runs=args.agent_max_runs,
                tempos=agent_tempos,
                base_use_crepe=args.use_crepe,
                prioritize_crepe=args.agent_prioritize_crepe,
                report_path=args.report,
                beat_times_override=beat_times_override,
            )
            raise SystemExit(exit_code)

        suite_result = run_suite(
            cases,
            use_crepe=args.use_crepe,
            output_dir=args.output_dir,
            threshold=args.threshold,
            report_path=args.report,
            tempo_override=args.tempo_bpm,
            beat_times_override=beat_times_override,
        )
        raise SystemExit(suite_result.exit_code)

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
