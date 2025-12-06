from __future__ import annotations

from typing import Any, Dict

from music21 import converter, midi

from .pipeline.models import AnalysisData
from .pipeline.stage_a import load_and_preprocess
from .pipeline.stage_b import extract_features
from .pipeline.stage_c import apply_theory
from .pipeline.stage_d import quantize_and_render


def _musicxml_to_midi_bytes(musicxml: str) -> bytes:
    """Convert a MusicXML string to MIDI bytes using music21."""

    score = converter.parseData(musicxml)
    midi_file = midi.translate.streamToMidiFile(score)
    return midi_file.writestr()


def transcribe_audio_pipeline(
    audio_path: str,
    use_crepe: bool = False,
    tempo_override: float | None = None,
    beat_times_override: list[float] | None = None,
    velocity_humanization: float | None = None,
    velocity_seed: int | None = None,
) -> Dict[str, Any]:
    """
    High-level API for the transcription pipeline (Stages A→D).

    Returns a dictionary containing the rendered MusicXML, MIDI bytes, metadata,
    and quantized note events for downstream consumers.
    """

    y, sr, meta = load_and_preprocess(audio_path)
    meta.tempo_override = tempo_override
    meta.tempo_bpm = tempo_override or meta.tempo_bpm
    meta.beat_times_override = beat_times_override

    timeline, notes, chords = extract_features(
        y,
        sr,
        meta,
        use_crepe=use_crepe,
        velocity_humanization=velocity_humanization,
        velocity_seed=velocity_seed,
    )

    analysis_data = AnalysisData(meta=meta, timeline=timeline, events=notes, chords=chords)
    events_with_theory = apply_theory(notes, analysis_data)

    musicxml_str = quantize_and_render(
        events_with_theory,
        analysis_data,
        tempo_override=tempo_override,
        beat_times_override=beat_times_override,
    )
    midi_bytes = _musicxml_to_midi_bytes(musicxml_str)

    return {
        "musicxml": musicxml_str,
        "midi_bytes": midi_bytes,
        "meta": meta,
        "notes": events_with_theory,
    }


# Legacy convenience wrapper
def transcribe_audio(
    audio_path: str,
    use_crepe: bool = False,
    tempo_override: float | None = None,
    beat_times_override: list[float] | None = None,
    velocity_humanization: float | None = None,
    velocity_seed: int | None = None,
) -> str:
    """Return only the MusicXML string for an audio file."""

    result = transcribe_audio_pipeline(
        audio_path,
        use_crepe=use_crepe,
        tempo_override=tempo_override,
        beat_times_override=beat_times_override,
        velocity_humanization=velocity_humanization,
        velocity_seed=velocity_seed,
    )
    return result["musicxml"]
