from __future__ import annotations

from typing import List, Tuple
import importlib

import librosa
import numpy as np

from .models import MetaData, FramePitch, NoteEvent, ChordEvent


def _crepe_available() -> bool:
    return importlib.util.find_spec("crepe") is not None


def _pitch_with_pyin(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _pitch_with_crepe(
    y: np.ndarray,
    sr: int,
    hop_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crepe = importlib.import_module("crepe")
    step_size_ms = hop_length * 1000.0 / sr
    if sr != 16000:
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr_used = 16000
    else:
        y_resampled = y
        sr_used = sr

    time, frequency, confidence, _ = crepe.predict(
        y_resampled, sr_used, step_size=step_size_ms, viterbi=True
    )
    voiced_flag = confidence > 0.3
    return time.astype(float), frequency.astype(float), voiced_flag, confidence.astype(float)


def _build_timeline(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
) -> List[FramePitch]:
    timeline: List[FramePitch] = []
    for t, hz, is_voiced, conf in zip(times, f0, voiced_flag, voiced_probs):
        if not is_voiced or hz is None or np.isnan(hz):
            timeline.append(FramePitch(time=float(t), pitch_hz=0.0, midi=None, confidence=float(conf)))
            continue
        midi = int(round(librosa.hz_to_midi(float(hz))))
        timeline.append(
            FramePitch(
                time=float(t),
                pitch_hz=float(hz),
                midi=midi,
                confidence=float(conf),
            )
        )
    return timeline


def _segment_notes_from_timeline(
    timeline: List[FramePitch],
    frame_duration: float,
    min_duration: float = 0.06,
    pitch_jump: float = 0.6,
) -> List[NoteEvent]:
    notes: List[NoteEvent] = []
    if not timeline:
        return notes

    start_time: float | None = None
    midi_values: List[int] = []
    conf_values: List[float] = []
    last_time: float | None = None

    for frame in timeline:
        if frame.midi is None:
            if start_time is not None and last_time is not None:
                end_time = frame.time
                duration = end_time - start_time
                if duration >= min_duration and midi_values:
                    midi_note = int(round(float(np.median(midi_values))))
                    pitch_hz = float(librosa.midi_to_hz(midi_note))
                    confidence = float(np.mean(conf_values)) if conf_values else 0.0
                    notes.append(
                        NoteEvent(
                            start_sec=start_time,
                            end_sec=end_time,
                            midi_note=midi_note,
                            pitch_hz=pitch_hz,
                            confidence=confidence,
                        )
                    )
                start_time = None
                midi_values = []
                conf_values = []
            last_time = frame.time
            continue

        if start_time is None:
            start_time = frame.time
            midi_values = [frame.midi]
            conf_values = [frame.confidence]
            last_time = frame.time
            continue

        current_median = float(np.median(midi_values))
        if abs(frame.midi - current_median) > pitch_jump:
            end_time = frame.time
            duration = end_time - start_time
            if duration >= min_duration and midi_values:
                midi_note = int(round(current_median))
                pitch_hz = float(librosa.midi_to_hz(midi_note))
                confidence = float(np.mean(conf_values)) if conf_values else 0.0
                notes.append(
                    NoteEvent(
                        start_sec=start_time,
                        end_sec=end_time,
                        midi_note=midi_note,
                        pitch_hz=pitch_hz,
                        confidence=confidence,
                    )
                )
            start_time = frame.time
            midi_values = [frame.midi]
            conf_values = [frame.confidence]
        else:
            midi_values.append(frame.midi)
            conf_values.append(frame.confidence)
        last_time = frame.time

    if start_time is not None:
        end_time = (last_time or start_time) + frame_duration
        duration = end_time - start_time
        if duration >= min_duration and midi_values:
            midi_note = int(round(float(np.median(midi_values))))
            pitch_hz = float(librosa.midi_to_hz(midi_note))
            confidence = float(np.mean(conf_values)) if conf_values else 0.0
            notes.append(
                NoteEvent(
                    start_sec=start_time,
                    end_sec=end_time,
                    midi_note=midi_note,
                    pitch_hz=pitch_hz,
                    confidence=confidence,
                )
            )

    return notes


def extract_features(
    y: np.ndarray,
    sr: int,
    meta: MetaData,
    use_crepe: bool = False,
) -> Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent]]:
    """
    Stage B: pitch tracking and simple note segmentation.
    """

    hop_length = meta.hop_length or 256
    fmin = librosa.note_to_hz("A1")
    fmax = librosa.note_to_hz("C7")

    if use_crepe and _crepe_available():
        times, f0, voiced_flag, voiced_probs = _pitch_with_crepe(y, sr, hop_length)
    elif _crepe_available():
        times, f0, voiced_flag, voiced_probs = _pitch_with_crepe(y, sr, hop_length)
    else:
        times, f0, voiced_flag, voiced_probs = _pitch_with_pyin(y, sr, hop_length, fmin=fmin, fmax=fmax)

    timeline = _build_timeline(times, f0, voiced_flag, voiced_probs)

    if len(timeline) >= 2:
        frame_duration = float(timeline[1].time - timeline[0].time)
    else:
        frame_duration = float(hop_length / sr)

    notes = _segment_notes_from_timeline(
        timeline,
        frame_duration=frame_duration,
        min_duration=0.06,
        pitch_jump=0.6,
    )

    chords: List[ChordEvent] = []
    return timeline, notes, chords
