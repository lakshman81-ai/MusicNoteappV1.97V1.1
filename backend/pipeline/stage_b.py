from __future__ import annotations

from typing import List, Tuple

import numpy as np
import librosa
import scipy.signal  # reserved for future use

from .models import (
    MetaData,
    FramePitch,
    NoteEvent,
    ChordEvent,
    AlternativePitch,
)

BASIC_PITCH_AVAILABLE = False  # explicitly disabled

try:
    import crepe

    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False


# --------- PITCH BACKENDS --------- #


def _pitch_via_pyin(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pitch tracking using librosa.pyin (monophonic F0).
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0, voiced_flag, voiced_probs


def _pitch_via_crepe(
    y: np.ndarray,
    sr: int,
    hop_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CREPE backend for F0. If installed, it usually performs better on piano.
    """
    step_size_ms = hop_length * 1000.0 / sr

    # CREPE expects 16kHz
    if sr != 16000:
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr_crepe = 16000
    else:
        y_16k = y
        sr_crepe = sr

    time, frequency, confidence, _ = crepe.predict(
        y_16k, sr_crepe, step_size=step_size_ms, viterbi=True
    )

    f0 = frequency.astype(float)
    voiced_flag = confidence > 0.3
    voiced_probs = confidence.astype(float)
    times = time.astype(float)
    return times, f0, voiced_flag, voiced_probs


# --------- TIMELINE BUILDING --------- #


def _build_timeline(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
) -> List[FramePitch]:
    from math import isnan

    timeline: List[FramePitch] = []
    for t, hz, vflag, vprob in zip(times, f0, voiced_flag, voiced_probs):
        if (not vflag) or hz is None or isnan(hz):
            timeline.append(
                FramePitch(
                    time=float(t),
                    pitch_hz=0.0,
                    midi=None,
                    confidence=float(vprob),
                )
            )
        else:
            midi = int(round(librosa.hz_to_midi(hz)))
            timeline.append(
                FramePitch(
                    time=float(t),
                    pitch_hz=float(hz),
                    midi=midi,
                    confidence=float(vprob),
                )
            )
    return timeline


# --------- SEGMENTATION (ONSET-DRIVEN) --------- #


def _segment_notes_from_timeline_onsets(
    timeline: List[FramePitch],
    onset_times: np.ndarray,
    total_duration: float,
    min_duration: float = 0.05,
) -> List[NoteEvent]:
    """
    Segment notes using onset times as primary boundaries.

    For each onset interval [t_k, t_{k+1}) we take the median MIDI pitch
    from the timeline within that window.
    """
    notes: List[NoteEvent] = []
    if not timeline:
        return notes

    times = np.array([f.time for f in timeline], dtype=float)
    midi_arr = np.array(
        [f.midi if f.midi is not None and f.midi > 0 else np.nan for f in timeline],
        dtype=float,
    )

    # Build boundaries: onset_0, onset_1, ..., final_time
    if onset_times.size == 0:
        # Fallback: treat entire clip as one region
        onset_times = np.array([0.0], dtype=float)

    boundaries = list(onset_times)
    if not boundaries or boundaries[0] > 0.01:
        boundaries.insert(0, 0.0)
    if total_duration > boundaries[-1] + 0.01:
        boundaries.append(total_duration)

    boundaries = np.array(boundaries, dtype=float)

    for i in range(len(boundaries) - 1):
        start_t = float(boundaries[i])
        end_t = float(boundaries[i + 1])
        dur = end_t - start_t
        if dur < min_duration:
            continue

        # Frames in this interval
        mask = (times >= start_t) & (times < end_t)
        if not np.any(mask):
            continue

        seg = midi_arr[mask]
        if np.all(np.isnan(seg)):
            continue

        midi_median = float(np.nanmedian(seg))
        midi_int = int(round(midi_median))
        hz = float(librosa.midi_to_hz(midi_int))

        notes.append(
            NoteEvent(
                start_sec=start_t,
                end_sec=end_t,
                midi_note=midi_int,
                pitch_hz=hz,
                confidence=1.0,
            )
        )

    return notes


# --------- MAIN STAGE B ENTRYPOINT --------- #


def extract_features(
    y: np.ndarray,
    sr: int,
    meta: MetaData,
    use_crepe: bool = False,
) -> Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent]]:
    """
    Stage B: Feature Extraction

    1. Pitch tracking (CREPE if available, else pyin)
    2. Build frame-wise pitch timeline
    3. Onset-based segmentation into notes
    4. Placeholder chord list
    """
    hop_length = getattr(meta, "hop_length", 512) or 512

    # Pitch range: piano-ish, but slightly generous
    fmin = librosa.note_to_hz("A1")  # 55 Hz
    fmax = librosa.note_to_hz("C7")  # 2093 Hz

    # Decide backend
    if use_crepe and CREPE_AVAILABLE:
        times, f0, voiced_flag, voiced_probs = _pitch_via_crepe(y, sr, hop_length)
    elif CREPE_AVAILABLE:
        # If CREPE is installed, prefer it
        times, f0, voiced_flag, voiced_probs = _pitch_via_crepe(y, sr, hop_length)
    else:
        times, f0, voiced_flag, voiced_probs = _pitch_via_pyin(
            y, sr, hop_length, fmin=fmin, fmax=fmax
        )

    # 1–2: timeline
    timeline = _build_timeline(times, f0, voiced_flag, voiced_probs)

    # 3: onset detection on the waveform
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    total_duration = float(len(y) / sr)

    notes = _segment_notes_from_timeline_onsets(
        timeline, onset_times, total_duration, min_duration=0.05
    )

    # Placeholder chords (front-end can run its own chord estimation)
    chords: List[ChordEvent] = []

    return timeline, notes, chords
