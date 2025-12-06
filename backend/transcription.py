from __future__ import annotations

import io
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
    buffer = io.BytesIO()
    midi_file.open(buffer)
    midi_file.write()
    midi_file.close()
    return buffer.getvalue()


def transcribe_audio_pipeline(audio_path: str, use_crepe: bool = False) -> Dict[str, Any]:
    """
    High-level API for the transcription pipeline (Stages A→D).

    Returns a dictionary containing the rendered MusicXML, MIDI bytes, metadata,
    and quantized note events for downstream consumers.
    """

    y, sr, meta = load_and_preprocess(audio_path)
    timeline, notes, chords = extract_features(y, sr, meta, use_crepe=use_crepe)

    analysis_data = AnalysisData(meta=meta, timeline=timeline, events=notes, chords=chords)
    events_with_theory = apply_theory(notes, analysis_data)

    musicxml_str = quantize_and_render(events_with_theory, analysis_data)
    midi_bytes = _musicxml_to_midi_bytes(musicxml_str)

    return {
        "musicxml": musicxml_str,
        "midi_bytes": midi_bytes,
        "meta": meta,
        "notes": events_with_theory,
    }


# Legacy convenience wrapper
def transcribe_audio(audio_path: str, use_crepe: bool = False) -> str:
    """Return only the MusicXML string for an audio file."""

    result = transcribe_audio_pipeline(audio_path, use_crepe=use_crepe)
    return result["musicxml"]
