from __future__ import annotations

from typing import List

from .models import NoteEvent, AnalysisData


def apply_theory(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
) -> List[NoteEvent]:
    """
    Stage C: placeholder theory layer.

    This implementation is intentionally non-destructive: it preserves the
    original timing and pitches while optionally attaching the events to the
    provided AnalysisData container.
    """

    analysis_data.events = events
    analysis_data.notes = events
    return events
