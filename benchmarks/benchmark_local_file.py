from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import sys

from music21 import converter, note as m21note

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backend.transcription import transcribe_audio_pipeline

NoteTuple = Tuple[int, float, float]  # (midi, onset_beats, duration_beats)


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
) -> float:
    if not reference:
        return 0.0
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
    return matched / len(reference)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local audio transcription against a reference MusicXML file."
    )
    parser.add_argument("audio_path", type=Path, help="Path to audio file (wav/mp3/etc.)")
    parser.add_argument("reference_path", type=Path, help="Path to reference MusicXML file")
    parser.add_argument("--use-crepe", action="store_true", help="Use CREPE for pitch tracking if available")
    args = parser.parse_args()

    if not args.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    if not args.reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference_path}")

    print(f"🎧 Audio: {args.audio_path}")
    print(f"🎼 Reference: {args.reference_path}")

    result = transcribe_audio_pipeline(str(args.audio_path), use_crepe=args.use_crepe)
    musicxml_text = result["musicxml"]
    midi_bytes: bytes = result["midi_bytes"]

    output_dir = Path("benchmarks_results")
    output_dir.mkdir(exist_ok=True)
    basename = args.audio_path.stem

    pred_xml_path = output_dir / f"{basename}_pred.musicxml"
    pred_mid_path = output_dir / f"{basename}_pred.mid"

    pred_xml_path.write_text(musicxml_text, encoding="utf-8")
    pred_mid_path.write_bytes(midi_bytes)

    ref_notes = extract_notes(args.reference_path)
    pred_notes = extract_notes(pred_xml_path)

    pitch_acc = match_accuracy(ref_notes, pred_notes, tol_beats=0.25, require_duration=False)
    rhythm_acc = match_accuracy(ref_notes, pred_notes, tol_beats=0.25, require_duration=True)

    print("\n=== BENCHMARK RESULT ===")
    print(f"Reference notes: {len(ref_notes)}")
    print(f"Predicted notes: {len(pred_notes)}")
    print(f"Pitch accuracy:  {pitch_acc * 100:.1f}%")
    print(f"Rhythm accuracy: {rhythm_acc * 100:.1f}% (±0.25 beats)")
    print(f"Predicted MusicXML saved to: {pred_xml_path}")
    print(f"Predicted MIDI saved to: {pred_mid_path}")


if __name__ == "__main__":
    main()
