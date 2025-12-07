from __future__ import annotations

import math
from typing import List, Optional, Tuple

import librosa
import numpy as np

from music21 import key as m21key, meter, metadata as m21meta, note as m21note, stream, tempo
from music21.musicxml.m21ToXml import GeneralObjectExporter

from .models import NoteEvent, AnalysisData, VexflowLayout, MetaData


SIXTEENTHS_PER_BEAT = 4.0
CANDIDATE_GRIDS = [4, 3, 5, 7]


def _parse_time_signature(time_signature: str) -> int:
    try:
        numerator = int(time_signature.split("/")[0])
        return max(numerator, 1)
    except Exception:
        return 4


def _determine_subdivisions(tempo_bpm: float, subdivisions_override: Optional[int] = None) -> int:
    if subdivisions_override and subdivisions_override > 0:
        return int(subdivisions_override)
    return int(SIXTEENTHS_PER_BEAT)


def _select_best_subdivision(
    events: List[NoteEvent],
    tempo_bpm: float,
    beat_times: List[float],
    candidates: List[int],
    time_signature: str,
    fallback: int,
) -> int:
    best_choice = fallback
    best_error = float("inf")

    for subdivisions in candidates:
        grid_times, grid_beats = _build_grid(events, tempo_bpm, beat_times, subdivisions)
        error = 0.0
        for event in events:
            start_quant, _ = _quantize_to_grid(event.start_sec, grid_times, grid_beats)
            end_quant, _ = _quantize_to_grid(event.end_sec, grid_times, grid_beats)
            error += (start_quant - event.start_sec) ** 2 + (end_quant - event.end_sec) ** 2
        # penalize grids that are too coarse for the meter
        if _parse_time_signature(time_signature) > 3 and subdivisions < 3:
            error *= 1.1
        if error < best_error:
            best_error = error
            best_choice = subdivisions

    return best_choice


def _estimate_tempo_and_beats(
    meta, tempo_override: float | None = None, beat_times_override: List[float] | None = None
) -> Tuple[float, List[float]]:
    explicit_tempo = tempo_override or getattr(meta, "tempo_override", None)
    explicit_beats = beat_times_override or getattr(meta, "beat_times_override", None)
    tempo_guess = explicit_tempo or meta.tempo_bpm or 120.0
    y = getattr(meta, "preprocessed_audio", None)
    sr = getattr(meta, "sample_rate", None)
    hop_length = getattr(meta, "hop_length", 256) or 256

    if explicit_beats:
        beats_sorted = sorted(float(b) for b in explicit_beats)
        if explicit_tempo is None and len(beats_sorted) >= 2:
            diffs = np.diff(np.asarray(beats_sorted))
            positive_diffs = diffs[diffs > 0]
            if positive_diffs.size:
                tempo_guess = 60.0 / float(np.median(positive_diffs))
        if tempo_guess <= 0:
            tempo_guess = meta.tempo_bpm or 120.0
        return tempo_guess, beats_sorted

    if y is None or sr is None:
        return tempo_guess, []

    try:
        tempo_bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        tempo_bpm = float(tempo_bpm)
        if tempo_bpm <= 0:
            tempo_bpm = tempo_guess
        return tempo_bpm, beat_times.tolist()
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
    events: List[NoteEvent],
    tempo_bpm: float,
    time_signature: str,
    beat_times: List[float],
    meta: Optional[MetaData] = None,
) -> None:
    beats_per_measure = _parse_time_signature(time_signature)
    subdivisions_override = getattr(meta, "quantization_subdivisions", None) if meta else None
    merge_gap_beats_override = getattr(meta, "quantization_merge_gap_beats", None) if meta else None

    base_subdivisions = _determine_subdivisions(tempo_bpm, subdivisions_override)
    candidate_grids = list(CANDIDATE_GRIDS)
    if subdivisions_override and subdivisions_override not in candidate_grids:
        candidate_grids.append(int(subdivisions_override))
    subdivisions = _select_best_subdivision(
        events, tempo_bpm, beat_times, candidate_grids, time_signature, base_subdivisions
    )
    grid_times, grid_beats = _build_grid(events, tempo_bpm, beat_times, subdivisions)

    beat_duration = 60.0 / tempo_bpm
    grid_unit_sec = beat_duration / float(subdivisions)
    grid_unit_beats = 1.0 / float(subdivisions)
    merge_gap_beats = merge_gap_beats_override if merge_gap_beats_override is not None else (1.0 / 8.0)
    merge_gap_sec = merge_gap_beats * beat_duration

    quantized_entries = []
    for event in sorted(events, key=lambda e: e.start_sec):
        start_quant, start_beats = _quantize_to_grid(event.start_sec, grid_times, grid_beats)
        end_quant, end_beats = _quantize_to_grid(event.end_sec, grid_times, grid_beats)

        duration_beats = max(end_beats - start_beats, grid_unit_beats)
        duration_sec = duration_beats * beat_duration
        min_end_sec = start_quant + grid_unit_sec
        end_sec = max(end_quant, min_end_sec, start_quant + duration_sec)
        duration_beats = max(duration_beats, (end_sec - start_quant) / beat_duration)
        end_beats = start_beats + duration_beats

        quantized_entries.append(
            {
                "event": event,
                "start_sec": start_quant,
                "end_sec": end_sec,
                "start_beats": start_beats,
                "end_beats": end_beats,
                "duration_beats": duration_beats,
            }
        )

    merged_entries = []
    for entry in quantized_entries:
        if (
            merged_entries
            and merged_entries[-1]["event"].midi_note == entry["event"].midi_note
            and entry["start_sec"] - merged_entries[-1]["end_sec"] < merge_gap_sec
        ):
            merged = merged_entries[-1]
            merged["end_sec"] = max(merged["end_sec"], entry["end_sec"])
            merged_duration_beats = (merged["end_sec"] - merged["start_sec"]) / beat_duration
            merged["duration_beats"] = max(merged_duration_beats, grid_unit_beats)
            merged["end_beats"] = merged["start_beats"] + merged["duration_beats"]
            continue

        merged_entries.append(entry)

    events.clear()
    for entry in merged_entries:
        duration_beats = max(entry["duration_beats"], grid_unit_beats)
        duration_sec = duration_beats * beat_duration
        end_sec = max(entry["end_sec"], entry["start_sec"] + duration_sec)
        duration_beats = max(duration_beats, (end_sec - entry["start_sec"]) / beat_duration)

        measure = int(entry["start_beats"] // beats_per_measure) + 1
        beat_in_measure = (entry["start_beats"] % beats_per_measure) + 1.0

        event = entry["event"]
        event.start_sec = entry["start_sec"]
        event.end_sec = end_sec
        event.measure = measure
        event.beat = beat_in_measure
        event.duration_beats = duration_beats
        events.append(event)


def _build_score(events: List[NoteEvent], tempo_bpm: float, time_signature: str, detected_key: str | None) -> stream.Score:
    score = stream.Score()
    score.insert(0, m21meta.Metadata())
    score.metadata.title = "Analyzed Audio"
    score.metadata.composer = "Music-Note-Creator"

    treble = stream.Part()
    treble.id = "P1"
    bass = stream.Part()
    bass.id = "P2"

    for part in (treble, bass):
        part.append(tempo.MetronomeMark(number=tempo_bpm))
        part.append(meter.TimeSignature(time_signature))
        if detected_key:
            part.append(m21key.Key(detected_key))

    for event in sorted(events, key=lambda e: e.start_sec):
        n = m21note.Note(event.midi_note)
        ql = float(event.duration_beats or 0.25)
        n.quarterLength = max(ql, 0.25)
        offset = float((event.measure - 1) * _parse_time_signature(time_signature) + ((event.beat or 1.0) - 1.0))
        target_part = bass if getattr(event, "voice", "voice1") == "voice2" else treble
        target_part.insert(offset, n)

    treble.makeMeasures(inPlace=True)
    bass.makeMeasures(inPlace=True)

    score.insert(0, treble)
    if len(bass.notes):
        score.insert(0, bass)
    return score


def quantize_and_render(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
    tempo_override: float | None = None,
    beat_times_override: List[float] | None = None,
) -> str:
    """Quantize NoteEvents and render them to a MusicXML string."""

    meta = analysis_data.meta
    tempo_bpm, beat_times = _estimate_tempo_and_beats(
        meta, tempo_override=tempo_override, beat_times_override=beat_times_override
    )
    meta.tempo_bpm = tempo_bpm
    time_signature = meta.time_signature or "4/4"

    if not beat_times:
        beat_duration = 60.0 / tempo_bpm
        max_event_end = max((e.end_sec for e in events), default=meta.duration_sec or 0.0)
        total_beats = max(1, int(math.ceil((max_event_end + beat_duration) / beat_duration)))
        beat_times = [i * beat_duration for i in range(total_beats + 1)]

    _quantize_events(
        events, tempo_bpm=tempo_bpm, time_signature=time_signature, beat_times=beat_times, meta=meta
    )
    analysis_data.events = events

    score = _build_score(events, tempo_bpm=tempo_bpm, time_signature=time_signature, detected_key=meta.detected_key)

    measures = []
    for part in score.parts:
        for m in part.getElementsByClass("Measure"):
            measures.append({"number": int(m.number), "part": part.id})
    analysis_data.vexflow_layout = VexflowLayout(measures=measures)

    exporter = GeneralObjectExporter()
    xml_obj = exporter.parse(score)
    if isinstance(xml_obj, (bytes, bytearray)):
        return xml_obj.decode("utf-8")
    return str(xml_obj)
