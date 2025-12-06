from __future__ import annotations

from typing import List

from music21 import (
    stream,
    note as m21note,
    meter,
    key as m21key,
    tempo as m21tempo,
    metadata as m21meta,
)
from music21.musicxml.m21ToXml import GeneralObjectExporter

from .models import NoteEvent, AnalysisData, VexflowLayout


def _quantize_notes_to_beats(
    events: List[NoteEvent],
    tempo_bpm: float,
    time_signature: str = "4/4",
) -> None:
    """
    Fill note.measure, note.beat, note.duration_beats.
    Very simple quantizer: snap start/end to nearest 1/16 note.
    """
    quarter_sec = 60.0 / tempo_bpm            # duration of 1 beat in seconds
    sixteenth_sec = quarter_sec / 4.0         # 1/16 note

    beats_per_measure = int(time_signature.split("/")[0])  # assumes x/4

    for e in events:
        # Quantize start & end in units of 1/16 notes
        n_start = round(e.start_sec / sixteenth_sec)
        n_end = round(e.end_sec / sixteenth_sec)
        if n_end <= n_start:
            n_end = n_start + 1  # ensure at least one 16th

        # Convert to beats (beat = quarter note)
        start_beats = n_start / 4.0
        end_beats = n_end / 4.0
        duration_beats = end_beats - start_beats

        measure = int(start_beats // beats_per_measure) + 1
        beat_in_measure = (start_beats % beats_per_measure) + 1.0

        e.measure = measure
        e.beat = beat_in_measure
        e.duration_beats = duration_beats


def _duration_beats_to_quarter_length(duration_beats: float) -> float:
    """
    Convert beats to music21 quarterLength.
    Under 4/4 we treat 1 beat = 1 quarter note.
    """
    return float(duration_beats)


def quantize_and_render(events: List[NoteEvent], analysis_data: AnalysisData) -> str:
    """
    Stage D: Quantize NoteEvent list and render to a MusicXML string.

    This is the function that transcription.py imports:
        from pipeline.stage_d import quantize_and_render
    """
    # --- 1. Get tempo & time signature from meta (with safe fallbacks) ---
    meta = getattr(analysis_data, "meta", None)
    tempo_bpm = 120.0
    time_sig = "4/4"

    if meta is not None:
        tempo_bpm = getattr(meta, "tempo_bpm", tempo_bpm) or tempo_bpm
        time_sig = getattr(meta, "time_signature", time_sig) or time_sig

    # --- 2. Quantize events in-place ---
    _quantize_notes_to_beats(events, tempo_bpm=tempo_bpm, time_signature=time_sig)

    # --- 3. Build a music21 Score/Part ---

    s = stream.Score()
    s.insert(0, m21meta.Metadata())
    s.metadata.title = "Transcription"
    s.metadata.composer = "Music-Note-Creator"

    part = stream.Part()
    part.id = "P1"

    # Time signature & tempo
    ts_num, ts_den = time_sig.split("/")
    part.append(m21tempo.MetronomeMark(number=tempo_bpm))
    part.append(meter.TimeSignature(f"{ts_num}/{ts_den}"))

    # Optional key signature if meta contains it
    detected_key = getattr(meta, "detected_key", None) if meta is not None else None
    if detected_key:
        part.append(m21key.Key(detected_key))

    # Insert notes in (measure, beat, start_sec) order
    def _sort_key(e: NoteEvent):
        return (
            getattr(e, "measure", 0) or 0,
            getattr(e, "beat", 0.0) or 0.0,
            getattr(e, "start_sec", 0.0) or 0.0,
        )

    for e in sorted(events, key=_sort_key):
        n = m21note.Note(e.midi_note)
        ql = _duration_beats_to_quarter_length(getattr(e, "duration_beats", 1.0) or 1.0)
        n.quarterLength = max(ql, 0.25)  # never shorter than a 16th
        part.append(n)

    s.insert(0, part)

    # --- 4. Vexflow layout info for frontend (optional but used elsewhere) ---
    measures = []
    for m in part.getElementsByClass("Measure"):
        measures.append(
            {
                "number": int(m.number),
                "width": 0,  # frontend decides actual width
            }
        )

    # Attach layout to analysis_data if it expects it
    try:
        analysis_data.vexflow_layout = VexflowLayout(measures=measures)
    except Exception:
        # If AnalysisData doesn't match, just skip layout
        pass

    # --- 5. Export to MusicXML string using GeneralObjectExporter ---
    exporter = GeneralObjectExporter()
    xml_obj = exporter.parse(s)

    if isinstance(xml_obj, (bytes, bytearray)):
        musicxml_str = xml_obj.decode("utf-8")
    else:
        musicxml_str = str(xml_obj)

    return musicxml_str
