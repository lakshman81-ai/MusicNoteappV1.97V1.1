from __future__ import annotations

from typing import List

from .models import NoteEvent, AnalysisData


def apply_theory(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
) -> List[NoteEvent]:
    """
    Stage C: music-theory post-processing.

    - Detects grace notes (short pickups into grid-aligned notes) and marks them
      as non-counting events.
    - Maps captured amplitude (0.0–1.0) to expressive dynamics.
    - Leaves enharmonic spelling to the key-aware MusicXML projection stage.
    """

    def _map_dynamic(amplitude: float) -> str:
        if amplitude < 0.25:
            return "p"
        if amplitude < 0.5:
            return "mp"
        if amplitude < 0.75:
            return "mf"
        return "f"

    events_sorted = sorted(events, key=lambda n: n.start_sec)
    for idx, note in enumerate(events_sorted):
        duration = float(note.end_sec - note.start_sec)
        if duration < 0.1 and idx + 1 < len(events_sorted):
            next_note = events_sorted[idx + 1]
            if next_note.start_sec - note.end_sec < 0.12:
                note.is_grace = True
        note.dynamic = _map_dynamic(getattr(note, "amplitude", 0.0))

    analysis_data.events = events_sorted
    analysis_data.notes = events_sorted
    return events_sorted
