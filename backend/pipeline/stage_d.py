from __future__ import annotations

import math
from typing import List, Tuple

import librosa
import numpy as np

from music21 import key as m21key, meter, metadata as m21meta, note as m21note, stream, tempo
from music21.musicxml.m21ToXml import GeneralObjectExporter

from .models import NoteEvent, AnalysisData, VexflowLayout


SIXTEENTHS_PER_BEAT = 4.0
TARGET_SUBDIVISION_SEC = 0.125


def _parse_time_signature(time_signature: str) -> int:
    try:
        numerator = int(time_signature.split("/")[0])
        return max(numerator, 1)
    except Exception:
        return 4


def _determine_subdivisions(tempo_bpm: float) -> int:
    beat_duration = 60.0 / tempo_bpm
    subdivisions = int(round(max(2.0, min(8.0, beat_duration / TARGET_SUBDIVISION_SEC))))
    return max(1, subdivisions)


def _estimate_tempo_and_beats(meta) -> Tuple[float, List[float]]:
    tempo_guess = meta.tempo_bpm or 120.0
    y = getattr(meta, "preprocessed_audio", None)
    sr = getattr(meta, "sample_rate", None)
    hop_length = getattr(meta, "hop_length", 256) or 256

    if y is None or sr is None:
        return tempo_guess, []

    try:
        tempo_bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        return float(tempo_bpm), beat_times.tolist()
    except Exception:
        return tempo_guess, []


def _build_grid(
    events: List[NoteEvent], tempo_bpm: float, beat_times: List[float], subdivisions: int
) -> Tuple[np.ndarray, np.ndarray]:
    beat_duration = 60.0 / tempo_bpm
    max_event_end = max((e.end_sec for e in events), default=0.0)
    grid_times: List[float] = []
    grid_beats: List[float] = []

    if beat_times:
        for idx, beat_time in enumerate(beat_times):
            next_time = beat_times[idx + 1] if idx + 1 < len(beat_times) else beat_time + beat_duration
            interval = (next_time - beat_time) / float(subdivisions)
            for sub in range(subdivisions):
                grid_times.append(beat_time + (sub * interval))
                grid_beats.append(idx + (sub / float(subdivisions)))
    else:
        total_beats = int(math.ceil((max_event_end + beat_duration) / beat_duration))
        for beat_idx in range(total_beats):
            base_time = beat_idx * beat_duration
            for sub in range(subdivisions):
                grid_times.append(base_time + (beat_duration * sub / float(subdivisions)))
                grid_beats.append(beat_idx + (sub / float(subdivisions)))

    if not grid_times:
        grid_times = [0.0]
        grid_beats = [0.0]

    return np.asarray(grid_times, dtype=float), np.asarray(grid_beats, dtype=float)


def _quantize_to_grid(time_value: float, grid_times: np.ndarray, grid_beats: np.ndarray) -> Tuple[float, float]:
    idx = int(np.argmin(np.abs(grid_times - time_value)))
    return float(grid_times[idx]), float(grid_beats[idx])


def _quantize_events(
    events: List[NoteEvent], tempo_bpm: float, time_signature: str, beat_times: List[float]
) -> None:
    beats_per_measure = _parse_time_signature(time_signature)
    subdivisions = _determine_subdivisions(tempo_bpm)
    grid_times, grid_beats = _build_grid(events, tempo_bpm, beat_times, subdivisions)

    for event in events:
        start_quant, start_beats = _quantize_to_grid(event.start_sec, grid_times, grid_beats)
        end_quant, end_beats = _quantize_to_grid(event.end_sec, grid_times, grid_beats)

        if end_quant <= start_quant:
            end_quant = start_quant + (60.0 / tempo_bpm / subdivisions)
            end_beats = start_beats + (1.0 / subdivisions)

        duration_beats = end_beats - start_beats

        measure = int(start_beats // beats_per_measure) + 1
        beat_in_measure = (start_beats % beats_per_measure) + 1.0

        event.start_sec = start_quant
        event.end_sec = end_quant
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
    tempo_bpm, beat_times = _estimate_tempo_and_beats(meta)
    meta.tempo_bpm = tempo_bpm
    time_signature = meta.time_signature or "4/4"

    _quantize_events(events, tempo_bpm=tempo_bpm, time_signature=time_signature, beat_times=beat_times)
    analysis_data.events = events

    score = _build_score(events, tempo_bpm=tempo_bpm, time_signature=time_signature, detected_key=meta.detected_key)

    measures = [{"number": int(m.number)} for m in score.parts[0].getElementsByClass("Measure")]
    analysis_data.vexflow_layout = VexflowLayout(measures=measures)

    exporter = GeneralObjectExporter()
    xml_obj = exporter.parse(score)
    if isinstance(xml_obj, (bytes, bytearray)):
        return xml_obj.decode("utf-8")
    return str(xml_obj)
