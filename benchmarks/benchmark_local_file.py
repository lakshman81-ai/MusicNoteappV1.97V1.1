from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from music21 import converter, note as m21note

# Make backend importable (assumes this file is in benchmarks/ at repo root)
BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
import sys
sys.path.append(str(BACKEND_ROOT))

from transcription import transcribe_audio_pipeline  # type: ignore


NoteTriple = Tuple[float, float, int]  # (start_beats, end_beats, midi_pitch)


def flatten_notes_m21(path: Path) -> List[NoteTriple]:
    """
    Load MIDI or MusicXML with music21 and return (start, end, pitch) in beats.
    """
    score = converter.parse(str(path))
    notes: List[NoteTriple] = []

    for n in score.recurse().notes:
        if isinstance(n, m21note.Note):
            start = float(n.offset)
            end = start + float(n.quarterLength)
            pitch = int(n.pitch.midi)
            notes.append((start, end, pitch))

    notes.sort(key=lambda x: x[0])
    return notes


def pitch_accuracy(ref: List[NoteTriple], pred: List[NoteTriple]) -> float:
    """
    Simple pitch accuracy: how many pitches match in order, ignoring timing.
    """
    if not ref or not pred:
        return 0.0
    L = min(len(ref), len(pred))
    correct = 0
    for i in range(L):
        if ref[i][2] == pred[i][2]:
            correct += 1
    return correct / L


def rhythm_accuracy(ref: List[NoteTriple], pred: List[NoteTriple], tol_beats: float = 0.25) -> float:
    """
    Rhythm accuracy: pitch must match AND duration within +/- tol_beats.
    """
    if not ref or not pred:
        return 0.0
    L = min(len(ref), len(pred))
    correct = 0
    for i in range(L):
        r_start, r_end, r_pitch = ref[i]
        p_start, p_end, p_pitch = pred[i]
        if r_pitch != p_pitch:
            continue
        r_dur = r_end - r_start
        p_dur = p_end - p_start
        if abs(r_dur - p_dur) <= tol_beats:
            correct += 1
    return correct / L


def main():
    parser = argparse.ArgumentParser(description="Benchmark one audio file against a reference score.")
    parser.add_argument("audio", type=Path, help="Path to audio file (wav/mp3)")
    parser.add_argument("reference", type=Path, help="Reference MIDI or MusicXML file")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds (audio)")
    parser.add_argument("--duration", type=float, default=None, help="Max duration in seconds (audio)")
    args = parser.parse_args()

    audio_path: Path = args.audio
    ref_path: Path = args.reference

    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        return
    if not ref_path.exists():
        print(f"[ERROR] Reference file not found: {ref_path}")
        return

    print(f"🎧 Audio:     {audio_path}")
    print(f"🎼 Reference: {ref_path}")
    print("Running transcription pipeline...")

    result = transcribe_audio_pipeline(
        str(audio_path),
        stereo_mode=False,
        use_mock=False,
        start_offset=args.start,
        max_duration=args.duration,
    )
    xml_text = result.musicxml

    # Save predicted MusicXML and convert to MIDI via music21
    out_dir = Path("benchmarks_results")
    out_dir.mkdir(exist_ok=True)
    base = audio_path.stem
    pred_xml_path = out_dir / f"{base}_pred.musicxml"
    pred_mid_path = out_dir / f"{base}_pred.mid"

    pred_xml_path.write_text(xml_text, encoding="utf-8")

    score_pred = converter.parse(xml_text)
    score_pred.write("midi", fp=str(pred_mid_path))

    # Flatten notes
    ref_notes = flatten_notes_m21(ref_path)
    pred_notes = flatten_notes_m21(pred_mid_path)

    pa = pitch_accuracy(ref_notes, pred_notes)
    ra = rhythm_accuracy(ref_notes, pred_notes, tol_beats=0.25)

    print("\n=== BENCHMARK RESULT ===")
    print(f"Reference notes: {len(ref_notes)}")
    print(f"Predicted notes: {len(pred_notes)}")
    print(f"Pitch accuracy:   {pa*100:.1f}%")
    print(f"Rhythm accuracy:  {ra*100:.1f}% (±0.25 beats)")
    print(f"Predicted MusicXML saved to: {pred_xml_path}")
    print(f"Predicted MIDI saved to:     {pred_mid_path}")


if __name__ == "__main__":
    main()
