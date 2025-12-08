from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

import librosa
import pretty_midi
import soundfile as sf
from music21 import converter, midi as m21midi

from backend.metrics import compute_metrics
from backend.pipeline.models import NoteEvent
from utils.audio_noise import add_noise_snr

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


def _midi_note_to_hz(midi_note: int) -> float:
    return float(librosa.midi_to_hz(midi_note)) if midi_note > 0 else 0.0


def _note_from_pretty_midi(note: pretty_midi.Note) -> NoteEvent:
    return NoteEvent(
        start_sec=float(note.start),
        end_sec=float(note.end),
        midi_note=int(note.pitch),
        pitch_hz=_midi_note_to_hz(int(note.pitch)),
    )


def load_reference_notes(path: Path) -> List[NoteEvent]:
    """Load reference notes from MIDI or MusicXML into NoteEvent structures."""

    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")

    if path.suffix.lower() in {".mid", ".midi"}:
        midi_data = pretty_midi.PrettyMIDI(str(path))
    else:
        score = converter.parse(str(path))
        midi_bytes = m21midi.translate.streamToMidiFile(score).writestr()
        midi_data = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))

    notes: List[NoteEvent] = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append(_note_from_pretty_midi(note))

    notes.sort(key=lambda n: n.start_sec)
    return notes


def load_predicted_notes(predicted: Iterable[NoteEvent]) -> List[NoteEvent]:
    return sorted(list(predicted), key=lambda n: n.start_sec)


def ensure_noisy_audio(source: Path, snr_db: float, target_root: Path) -> Path:
    """Create (or load) a noisy copy of the source audio at the requested SNR."""

    target_dir = target_root / f"SNR{int(snr_db)}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{source.stem}_SNR{int(snr_db)}.wav"
    if target_path.exists():
        return target_path

    audio, sr = librosa.load(str(source), sr=None, mono=False)
    seed = abs(hash((source.as_posix(), snr_db))) % (2**32)
    noisy = add_noise_snr(audio, snr_db, seed=seed)
    sf.write(target_path, noisy.T if noisy.ndim > 1 else noisy, sr)
    return target_path


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
    if not case.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {case.audio_path}")
    if not case.reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {case.reference_path}")

    case_dir = output_dir / case.scenario
    case_dir.mkdir(parents=True, exist_ok=True)

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

    pred_xml_path = case_dir / f"{case.slug}_pred.musicxml"
    pred_mid_path = case_dir / f"{case.slug}_pred.mid"

    pred_xml_path.write_text(musicxml_text, encoding="utf-8")
    pred_mid_path.write_bytes(midi_bytes)

    ref_notes = load_reference_notes(case.reference_path)
    pred_notes = load_predicted_notes(result.get("notes", []))

    metrics = compute_metrics(ref_notes, pred_notes)
    metrics_path = case_dir / f"{case.audio_path.stem}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

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
        "predicted_musicxml": str(pred_xml_path),
        "predicted_midi": str(pred_mid_path),
        "metrics_path": str(metrics_path),
        **metrics,
    }

    print("=== BENCHMARK RESULT ===")
    print(f"Reference notes: {len(ref_notes)}")
    print(f"Predicted notes: {len(pred_notes)}")
    print("Metric snapshot:")
    for key in sorted(metrics.keys()):
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"Predicted MusicXML saved to: {pred_xml_path}")
    print(f"Predicted MIDI saved to: {pred_mid_path}")
    print(f"Metrics saved to: {metrics_path}")

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

    if not summaries:
        raise ValueError("No benchmark cases provided to run_suite")

    metric_keys = [k for k in summaries[0].keys() if k.isupper() or "F" in k or "Precision" in k or "Recall" in k]

    def _avg(key: str, items: List[Dict[str, float | int | str]]) -> float:
        return sum(float(item.get(key, 0.0)) for item in items) / float(len(items))

    suite_summary: Dict[str, Dict[str, object]] = {}
    for scenario in {s["scenario"] for s in summaries}:
        scenario_cases = [s for s in summaries if s["scenario"] == scenario]
        metrics_avg = {key: _avg(key, scenario_cases) for key in metric_keys}
        scenario_summary = {
            "benchmark": scenario,
            "case_count": len(scenario_cases),
            "metrics": metrics_avg,
            "threshold": threshold,
            "strategy_label": strategy_label or "default",
            "cases": scenario_cases,
        }
        scenario_dir = output_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        summary_path = scenario_dir / "suite_summary.json"
        summary_path.write_text(json.dumps(scenario_summary, indent=2, sort_keys=True), encoding="utf-8")
        suite_summary[scenario] = scenario_summary
        print(f"Scenario {scenario} summary saved to: {summary_path}")

    all_hm = [_avg("HM", s["cases"]) for s in suite_summary.values() if "HM" in metric_keys]
    overall_hm = sum(all_hm) / len(all_hm)
    exit_code = 0 if overall_hm >= threshold else 1
    if exit_code:
        print(
            f"Overall harmonic mean {overall_hm:.3f} below threshold {threshold:.3f}; failing with non-zero exit code."
        )

    if report_path:
        write_markdown_report(suite_summary, report_path)
        print(f"Markdown report saved to: {report_path}")

    return SuiteRunResult(exit_code=exit_code, summary=suite_summary, strategy_label=strategy_label)


def write_markdown_report(suite_summary: Dict[str, object], report_path: Path) -> None:
    header = [
        "# Benchmark metric snapshot",
        "",
        "Aggregated metrics from `benchmarks/benchmark_local_file.py --suite`.",
        "",
        "| Scenario | HM | RPA | CA | OA | GEA | OnsetF | OffsetF | OnsetOffsetF |",
        "|----------|----|-----|----|----|-----|--------|---------|-------------|",
    ]

    rows: List[str] = []
    for scenario, summary in suite_summary.items():
        metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
        rows.append(
            "| "
            + " | ".join(
                [
                    scenario,
                    f"{metrics.get('HM', 0):.3f}",
                    f"{metrics.get('RPA', 0):.3f}",
                    f"{metrics.get('CA', 0):.3f}",
                    f"{metrics.get('OA', 0):.3f}",
                    f"{metrics.get('GEA', 0):.3f}",
                    f"{metrics.get('OnsetF', 0):.3f}",
                    f"{metrics.get('OffsetF', 0):.3f}",
                    f"{metrics.get('OnsetOffsetF', 0):.3f}",
                ]
            )
            + " |"
        )

    report_path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


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
        scenario_metrics = [s.get("metrics", {}) for s in summary.values()]
        average_hm = sum(m.get("HM", 0.0) for m in scenario_metrics) / max(len(scenario_metrics), 1)
        average_onset_f = sum(m.get("OnsetF", 0.0) for m in scenario_metrics) / max(len(scenario_metrics), 1)
        history.append(
            {
                "iteration": idx,
                "strategy": strategy.describe(),
                "average_hm": average_hm,
                "average_onset_f": average_onset_f,
                "report_path": iter_report_path.as_posix() if iter_report_path else None,
                "exit_code": suite_result.exit_code,
            }
        )

        hm_ok = average_hm >= target_accuracy
        if hm_ok:
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


def get_noise_robustness_cases(
    base_cases: List[BenchmarkCase], snr_levels: Iterable[int] = (20, 10, 0)
) -> Dict[str, List[BenchmarkCase]]:
    noise_root = REPO_ROOT / "benchmarks" / "05_noise_robustness"
    suites: Dict[str, List[BenchmarkCase]] = {}
    for snr_db in snr_levels:
        cases: List[BenchmarkCase] = []
        for case in base_cases:
            noisy_audio = ensure_noisy_audio(case.audio_path, float(snr_db), noise_root)
            cases.append(
                BenchmarkCase(
                    name=f"{case.name} (SNR{snr_db})",
                    audio_path=noisy_audio,
                    reference_path=case.reference_path,
                    scenario=f"noise_robustness/SNR{snr_db}",
                )
            )
        suites[f"SNR{snr_db}"] = cases
    return suites


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
