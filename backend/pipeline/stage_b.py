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


def _pitch_with_yin(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f0 = librosa.yin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    voiced_flag = np.isfinite(f0) & (f0 > 0)
    voiced_probs = np.where(voiced_flag, 0.5, 0.0)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0, voiced_flag, voiced_probs


def _pitch_with_crepe(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
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


def _harmonic_summation_refine(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    times: np.ndarray,
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    n_fft: int = 2048,
    harmonics: int = 4,
) -> np.ndarray:
    """Refine F0 using harmonic summation around the detected pitch."""

    if not np.any(voiced_mask):
        return f0

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    refined = np.copy(f0)
    for idx, (time, base_f0, is_voiced) in enumerate(zip(times, f0, voiced_mask)):
        if not is_voiced or base_f0 is None or np.isnan(base_f0) or base_f0 <= 0:
            continue

        frame_idx = int(
            np.clip(
                np.round(librosa.time_to_frames(time, sr=sr, hop_length=hop_length)),
                0,
                S.shape[1] - 1,
            )
        )

        candidate_freqs = base_f0 * (2 ** np.linspace(-0.25, 0.25, 5))
        best_freq = base_f0
        best_score = -np.inf

        for candidate in candidate_freqs:
            if candidate <= 0:
                continue

            score = 0.0
            for h in range(1, harmonics + 1):
                freq = candidate * h
                if freq > freqs[-1]:
                    break
                mag = float(np.interp(freq, freqs, S[:, frame_idx]))
                score += mag / h

            if score > best_score:
                best_score = score
                best_freq = candidate

        refined[idx] = best_freq

    return refined


def _apply_voicing_hysteresis(
    voiced_probs: np.ndarray, on_threshold: float = 0.55, off_threshold: float = 0.35
) -> np.ndarray:
    """Convert probabilities into a stable voiced mask using hysteresis."""

    mask = np.zeros_like(voiced_probs, dtype=bool)
    active = False
    for i, prob in enumerate(voiced_probs):
        if active:
            if prob <= off_threshold:
                active = False
        else:
            if prob >= on_threshold:
                active = True
        mask[i] = active
    return mask


def _smooth_midi_with_voicing(
    f0: np.ndarray,
    voiced_probs: np.ndarray,
    prob_threshold: float = 0.45,
    window: int = 3,
) -> np.ndarray:
    midi = librosa.hz_to_midi(f0)
    midi = np.where(np.isfinite(midi), midi, np.nan)
    smoothed = midi.copy()

    half = window // 2
    for i in range(len(midi)):
        if np.isnan(midi[i]) or voiced_probs[i] < prob_threshold:
            continue
        start = max(0, i - half)
        end = min(len(midi), i + half + 1)

        window_mask = (~np.isnan(midi[start:end])) & (voiced_probs[start:end] >= prob_threshold)
        if not np.any(window_mask):
            continue

        smoothed[i] = float(np.median(midi[start:end][window_mask]))

    return smoothed


def _build_timeline(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
    midi_values: np.ndarray | None,
    rms: np.ndarray | None,
    min_confidence: float,
    rms_floor: float | None,
) -> List[FramePitch]:
    timeline: List[FramePitch] = []
    for idx, (t, hz, is_voiced, conf) in enumerate(
        zip(times, f0, voiced_flag, voiced_probs)
    ):
        if not is_voiced or hz is None or np.isnan(hz):
            timeline.append(FramePitch(time=float(t), pitch_hz=0.0, midi=None, confidence=float(conf)))
            continue
        midi_value = midi_values[idx] if midi_values is not None else librosa.hz_to_midi(float(hz))
        if np.isnan(midi_value):
            timeline.append(FramePitch(time=float(t), pitch_hz=0.0, midi=None, confidence=float(conf)))
            continue
        midi = int(round(float(midi_value)))
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

    if _crepe_available():
        times, f0, voiced_flag, voiced_probs = _pitch_with_crepe(y, sr, hop_length)
    else:
        try:
            times, f0, voiced_flag, voiced_probs = _pitch_with_pyin(
                y, sr, hop_length, fmin=fmin, fmax=fmax
            )
        except Exception:
            times, f0, voiced_flag, voiced_probs = _pitch_with_yin(
                y, sr, hop_length, fmin=fmin, fmax=fmax
            )

    voiced_mask = _apply_voicing_hysteresis(voiced_probs) & voiced_flag
    refined_f0 = _harmonic_summation_refine(
        y, sr, hop_length, times, f0, voiced_mask, n_fft=2048, harmonics=4
    )
    midi_smoothed = _smooth_midi_with_voicing(refined_f0, voiced_probs)

    timeline = _build_timeline(
        times,
        refined_f0,
        voiced_mask,
        voiced_probs,
        midi_smoothed,
        rms=None,
        min_confidence=0.0,
        rms_floor=None,
    )

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
