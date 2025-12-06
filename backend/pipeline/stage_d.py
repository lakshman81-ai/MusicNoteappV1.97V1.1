from __future__ import annotations

from typing import List

from music21 import key as m21key, meter, metadata as m21meta, note as m21note, stream, tempo
from music21.musicxml.m21ToXml import GeneralObjectExporter

from .models import NoteEvent, AnalysisData, VexflowLayout


SIXTEENTHS_PER_BEAT = 4.0


def _parse_time_signature(time_signature: str) -> int:
    try:
        numerator = int(time_signature.split("/")[0])
        return max(numerator, 1)
    except Exception:
        return 4


def _quantize_events(events: List[NoteEvent], tempo_bpm: float, time_signature: str) -> None:
    quarter_sec = 60.0 / tempo_bpm
    sixteenth_sec = quarter_sec / SIXTEENTHS_PER_BEAT
    beats_per_measure = _parse_time_signature(time_signature)

    for event in events:
        start_quant = round(event.start_sec / sixteenth_sec) * sixteenth_sec
        end_quant = round(event.end_sec / sixteenth_sec) * sixteenth_sec
        if end_quant <= start_quant:
            end_quant = start_quant + sixteenth_sec

        start_beats = start_quant / quarter_sec
        end_beats = end_quant / quarter_sec
        duration_beats = end_beats - start_beats

        measure = int(start_beats // beats_per_measure) + 1
        beat_in_measure = (start_beats % beats_per_measure) + 1.0

        event.measure = measure
        event.beat = beat_in_measure
        event.duration_beats = duration_beats


def _build_score(events: List[NoteEvent], tempo_bpm: float, time_signature: str, detected_key: str | None) -> stream.Score:
    score = stream.Score()
    score.insert(0, m21meta.Metadata())
    score.metadata.title = "Analyzed Audio"
    score.metadata.composer = "Music-Note-Creator"

    part = stream.Part()
    part.id = "P1"
    part.append(tempo.MetronomeMark(number=tempo_bpm))
    part.append(meter.TimeSignature(time_signature))

    if detected_key:
        part.append(m21key.Key(detected_key))

    for event in sorted(events, key=lambda e: e.start_sec):
        n = m21note.Note(event.midi_note)
        ql = float(event.duration_beats or 0.25)
        n.quarterLength = max(ql, 0.25)
        offset = float((event.measure - 1) * _parse_time_signature(time_signature) + ((event.beat or 1.0) - 1.0))
        part.insert(offset, n)

    part.makeMeasures(inPlace=True)
    score.insert(0, part)
    return score


def quantize_and_render(events: List[NoteEvent], analysis_data: AnalysisData) -> str:
    """Quantize NoteEvents and render them to a MusicXML string."""

    meta = analysis_data.meta
    tempo_bpm = meta.tempo_bpm or 120.0
    time_signature = meta.time_signature or "4/4"

    _quantize_events(events, tempo_bpm=tempo_bpm, time_signature=time_signature)
    analysis_data.events = events

    score = _build_score(events, tempo_bpm=tempo_bpm, time_signature=time_signature, detected_key=meta.detected_key)

    measures = [{"number": int(m.number)} for m in score.parts[0].getElementsByClass("Measure")]
    analysis_data.vexflow_layout = VexflowLayout(measures=measures)

    exporter = GeneralObjectExporter()
    xml_obj = exporter.parse(score)
    if isinstance(xml_obj, (bytes, bytearray)):
        return xml_obj.decode("utf-8")
    return str(xml_obj)
