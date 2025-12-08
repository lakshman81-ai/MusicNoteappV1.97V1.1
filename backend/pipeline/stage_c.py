from __future__ import annotations

from typing import List

from backend.config_manager import get_config
from .models import AnalysisData, NoteEvent


def _stage_c_params() -> dict:
    cfg = get_config()
    ppq = int(cfg.get("ppq", 120))
    bar_ticks = int(cfg.get("bar_ticks", 4 * ppq))
    return {
        "ppq": ppq,
        "bar_ticks": bar_ticks,
        "min_note_ticks": int(cfg.get("min_note_ticks", 30)),
        "min_grid_ms": float(cfg.get("min_grid_ms", 4.0)),
    }


def apply_theory(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
) -> List[NoteEvent]:
    """
    Stage C: deterministic quantization and theory post-processing.

    - Snaps onsets/offsets to a 4 ms grid (PPQ=120 → 4.166… ms at 120 BPM).
    - Enforces pitch range (C1–C8), non-overlap per pitch, and min durations.
    - Flags staccato (<30 ticks) and grace pickup notes (<120 ms pre-roll).
    - Assigns measure/beat using a 480-tick bar (4/4) and writes tick metadata.
    """

    params = _stage_c_params()

    tempo_bpm = float(analysis_data.meta.tempo_bpm or 120.0)
    seconds_per_tick = 60.0 / (tempo_bpm * params["ppq"])
    min_grid_ticks = max(1, int(round((params["min_grid_ms"] / 1000.0) / seconds_per_tick)))

    def _snap_to_grid(seconds: float) -> int:
        raw_ticks = seconds / seconds_per_tick
        snapped = int(round(raw_ticks / min_grid_ticks) * min_grid_ticks)
        return max(snapped, 0)

    def _map_dynamic(amplitude: float) -> str:
        if amplitude < 0.25:
            return "p"
        if amplitude < 0.5:
            return "mp"
        if amplitude < 0.75:
            return "mf"
        return "f"

    events_sorted = sorted(events, key=lambda n: n.start_sec)
    processed: List[NoteEvent] = []
    last_end_per_pitch: dict[int, int] = {}

    for idx, note in enumerate(events_sorted):
        start_tick = _snap_to_grid(note.start_sec)
        end_tick = max(_snap_to_grid(note.end_sec), start_tick + 1)
        duration_ticks = max(end_tick - start_tick, 1)

        midi_note = int(note.midi_note)
        midi_note = min(max(midi_note, 24), 108)

        previous_end = last_end_per_pitch.get(midi_note)
        if previous_end is not None and start_tick < previous_end:
            start_tick = previous_end
            end_tick = max(end_tick, start_tick + 1)
            duration_ticks = end_tick - start_tick

        duration_seconds = duration_ticks * seconds_per_tick
        if duration_ticks < params["min_note_ticks"] and duration_seconds < 0.12:
            note.is_grace = True
        note.articulation = "staccato" if duration_ticks < params["min_note_ticks"] else note.articulation

        note.dynamic = _map_dynamic(getattr(note, "amplitude", 0.0))
        note.midi_note = midi_note
        note.start_tick = start_tick
        note.duration_ticks = duration_ticks
        note.start_sec = start_tick * seconds_per_tick
        note.end_sec = note.start_sec + duration_ticks * seconds_per_tick

        measure_idx = start_tick // params["bar_ticks"]
        beat_within = (start_tick % params["bar_ticks"]) / float(params["ppq"])
        note.measure = int(measure_idx + 1)
        note.beat = float(beat_within + 1.0)
        note.duration_beats = float(duration_ticks / params["ppq"])

        processed.append(note)
        last_end_per_pitch[midi_note] = end_tick

    analysis_data.events = processed
    analysis_data.notes = processed
    return processed
