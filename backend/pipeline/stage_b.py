from __future__ import annotations

from typing import List, Tuple
import copy
import importlib
import os
import tempfile
from pathlib import Path
import warnings

NUMBA_CACHE = Path(tempfile.gettempdir()) / "numba_cache"
os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
# Keep numba disabled to prevent heavy compilation during librosa.pyin/sequence usage.
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numba  # noqa: E402

numba.config.DISABLE_JIT = True
numba.config.CACHE_DIR = str(NUMBA_CACHE)


def _noop_decorator(*_args, **_kwargs):
    def wrapper(func):
        return func

    return wrapper


numba.guvectorize = _noop_decorator  # type: ignore[attr-defined]
numba.vectorize = _noop_decorator  # type: ignore[attr-defined]
numba.jit = _noop_decorator  # type: ignore[attr-defined]
numba.njit = _noop_decorator  # type: ignore[attr-defined]

import librosa
import numpy as np
from scipy.signal import medfilt

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
    if os.environ.get("DISABLE_PYIN", "1") == "1":
        return _pitch_with_spectral_peaks(y, sr, hop_length, fmin=fmin, fmax=fmax)

    cache_dir = Path(tempfile.gettempdir()) / "numba_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(f"librosa.pyin failed ({exc}); falling back to spectral peak tracker", RuntimeWarning)
        return _pitch_with_spectral_peaks(y, sr, hop_length, fmin=fmin, fmax=fmax)

    finite = np.sum(~np.isnan(f0))
    if finite < len(f0) * 0.5:
        return _pitch_with_spectral_peaks(y, sr, hop_length, fmin=fmin, fmax=fmax)

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0, voiced_flag, voiced_probs


def _pitch_with_spectral_peaks(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S = np.abs(
        librosa.stft(y, n_fft=2048, hop_length=hop_length, window="hann", center=True, pad_mode="reflect")
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    band_freqs = freqs[band_mask]

    pitch_track = np.zeros(S.shape[1])
    conf_track = np.zeros_like(pitch_track)

    for i in range(S.shape[1]):
        mag_col = S[:, i]
        band_mag = mag_col[band_mask]
        if band_mag.size == 0:
            continue
        idx = int(np.argmax(band_mag))
        pitch_track[i] = float(band_freqs[idx])
        conf_track[i] = float(band_mag[idx])

    if np.max(conf_track) > 0:
        conf_track = conf_track / float(np.max(conf_track))

    voiced_flag = pitch_track > 0
    times = librosa.times_like(pitch_track, sr=sr, hop_length=hop_length)
    return times, pitch_track, voiced_flag, conf_track


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
    rms: np.ndarray | None,
    min_confidence: float,
    rms_floor: float | None,
) -> List[FramePitch]:
    timeline: List[FramePitch] = []
    for idx, (t, hz, is_voiced, conf) in enumerate(zip(times, f0, voiced_flag, voiced_probs)):
        energy = rms[idx] if rms is not None and idx < len(rms) else None
        if (
            hz is None
            or np.isnan(hz)
            or not np.isfinite(conf)
            or conf < min_confidence
            or (rms_floor is not None and (energy is None or energy < rms_floor))
        ):
            timeline.append(
                FramePitch(
                    time=float(t),
                    pitch_hz=0.0,
                    midi=None,
                    confidence=float(conf) if np.isfinite(conf) else 0.0,
                )
            )
            continue
        midi = int(round(librosa.hz_to_midi(float(hz))))
        timeline.append(
            FramePitch(
                time=float(t),
                pitch_hz=float(hz),
                midi=midi,
                confidence=float(conf) if np.isfinite(conf) else 0.0,
            )
        )
    return timeline


def _segment_notes_from_timeline(
    timeline: List[FramePitch],
    frame_duration: float,
    min_duration: float = 0.04,
    pitch_jump: float = 0.6,
    min_gap: float = 0.0,
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
                if min_gap > 0.0 and (end_time - last_time) <= min_gap:
                    last_time = frame.time
                    continue
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


def _post_process_notes(notes: List[NoteEvent]) -> List[NoteEvent]:
    if not notes:
        return notes

    median_midi = float(np.median([n.midi_note for n in notes]))
    preferred_mid = median_midi
    while preferred_mid > 76:
        preferred_mid -= 12
    while preferred_mid < 48:
        preferred_mid += 12

    target_center = 60.0
    while preferred_mid - target_center > 4:
        preferred_mid -= 12
    while target_center - preferred_mid > 4:
        preferred_mid += 12

    for n in notes:
        while n.midi_note - preferred_mid > 4:
            n.midi_note -= 12
            n.pitch_hz = float(librosa.midi_to_hz(n.midi_note))
        while preferred_mid - n.midi_note > 4:
            n.midi_note += 12
            n.pitch_hz = float(librosa.midi_to_hz(n.midi_note))

    merged: List[NoteEvent] = []
    for note in notes:
        if merged and merged[-1].midi_note == note.midi_note and note.start_sec - merged[-1].end_sec < 0.1:
            merged[-1].end_sec = note.end_sec
            merged[-1].confidence = float((merged[-1].confidence + note.confidence) / 2)
        else:
            merged.append(note)

    return merged


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
    # Broader range for mixed mp3s (voices + simple instruments)
    fmin = librosa.note_to_hz("A1")
    fmax = librosa.note_to_hz("C6")

    if use_crepe and _crepe_available():
        times, f0, voiced_flag, voiced_probs = _pitch_with_crepe(y, sr, hop_length)
    else:
        times, f0, voiced_flag, voiced_probs = _pitch_with_pyin(
            y, sr, hop_length, fmin=fmin, fmax=fmax
        )

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True).flatten()
    rms_floor = None if not rms.size else float(np.percentile(rms, 5))
    min_confidence = 0.03

    timeline = _build_timeline(
        times,
        f0,
        voiced_flag,
        voiced_probs,
        rms=rms,
        min_confidence=min_confidence,
        rms_floor=rms_floor,
    )

    midi_series = np.array([fp.midi if fp.midi is not None else np.nan for fp in timeline], dtype=float)
    voice_mask = np.array(
        [
            (fp.midi is not None)
            and (fp.confidence >= min_confidence)
            and (rms is None or idx >= len(rms) or rms[idx] >= (rms_floor or 0.0))
            for idx, fp in enumerate(timeline)
        ],
        dtype=bool,
    )

    if np.any(np.isfinite(midi_series)):
        valid_idx = np.flatnonzero(np.isfinite(midi_series))
        interp = np.interp(np.arange(len(midi_series)), valid_idx, midi_series[valid_idx])
        smoothed = medfilt(interp, kernel_size=5)

        # Fill short gaps (1–2 frames) to avoid choppy segmentation, but keep longer rests silent.
        gap_kernel = np.ones(3, dtype=int)
        padded = np.pad(voice_mask.astype(int), (1, 1), mode="edge")
        closed = np.convolve(padded, gap_kernel, mode="valid") >= 2

        for idx, (fp, midi_val) in enumerate(zip(timeline, smoothed)):
            if closed[idx] and np.isfinite(midi_val):
                midi_int = int(round(float(midi_val)))
                fp.midi = midi_int
                fp.pitch_hz = float(librosa.midi_to_hz(midi_int))
            else:
                fp.midi = None
                fp.pitch_hz = 0.0

    if len(timeline) >= 2:
        frame_duration = float(timeline[1].time - timeline[0].time)
    else:
        frame_duration = float(hop_length / sr)

    raw_notes = _segment_notes_from_timeline(
        timeline,
        frame_duration=frame_duration,
        min_duration=0.03,
        pitch_jump=0.85,
        min_gap=frame_duration * 2.5,
    )

    notes = _post_process_notes(raw_notes)
    notes = _post_process_notes(copy.deepcopy(notes))

    chords: List[ChordEvent] = []
    return timeline, notes, chords
