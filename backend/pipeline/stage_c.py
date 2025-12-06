from __future__ import annotations

from typing import List

from .models import NoteEvent, AnalysisData


def apply_theory(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
) -> List[NoteEvent]:
    """
    Stage C: Minimal theory layer.

    The original pipeline calls:
        events_with_theory = apply_theory(notes, analysis_data)

    For benchmarking, we keep this a no-op on the notes:
    - Do NOT change pitches, onsets, or durations.
    - Just make sure analysis_data keeps a reference to the note list.
    - Return the events list unchanged.
    """
    try:
        # Keep notes attached to the analysis object, in case Stage D or
        # anything else wants to read them later.
        analysis_data.notes = events
    except Exception:
        # If AnalysisData is not the type we expect, just ignore.
        pass

    # No modifications for now – this keeps transcription pure.
    return events
