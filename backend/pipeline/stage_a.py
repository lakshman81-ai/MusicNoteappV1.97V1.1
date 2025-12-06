from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .models import MetaData


DEFAULT_TARGET_SR = 22050
DEFAULT_HOP_LENGTH = 256
TARGET_LUFS = -20.0


def _load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load audio from disk preserving the native sample rate."""

    path_str = str(audio_path)
    try:
        data, sr = sf.read(path_str, always_2d=False)
    except Exception:
        data, sr = librosa.load(path_str, sr=None, mono=False)
    y = np.asarray(data, dtype=np.float32)
    return y, int(sr)


def _to_mono(y: np.ndarray) -> np.ndarray:
    """Downmix stereo to mono by averaging channels."""

    if y.ndim == 1:
        return y.astype(np.float32)
    return np.mean(y, axis=1, dtype=np.float32)


def _trim_silence(y: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """Trim leading and trailing silence using an energy threshold."""

    try:
        non_silent_indices = librosa.effects.split(y, top_db=top_db)
    except Exception:
        return y.astype(np.float32)

    if non_silent_indices.size == 0:
        return np.array([], dtype=np.float32)

    start, end = non_silent_indices[0][0], non_silent_indices[-1][1]
    return y[start:end].astype(np.float32)


def _resample(y: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    """Resample waveform to the target sample rate."""

    if sr == target_sr:
        return y.astype(np.float32), sr
    y_resampled = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return y_resampled.astype(np.float32), target_sr


def _normalize_loudness(y: np.ndarray, sr: int) -> Tuple[np.ndarray, float | None]:
    """Normalize loudness to approximately TARGET_LUFS with a peak fallback."""

    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        y_norm = pyln.normalize.loudness(y, loudness, TARGET_LUFS)
        return y_norm.astype(np.float32), float(loudness)
    except Exception:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            return (y / peak).astype(np.float32), None
        return y.astype(np.float32), None


def _validate_signal(y: np.ndarray, rms_floor_db: float = -50.0, peak_floor: float = 1e-4) -> Tuple[float, float]:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    rms_db = 20.0 * np.log10(max(rms, 1e-12))
    if peak < peak_floor or rms_db < rms_floor_db:
        raise ValueError(
            f"Audio lacks usable signal after normalization (peak={peak:.5f}, rms_db={rms_db:.1f})"
        )
    return rms_db, peak


def load_and_preprocess(
    audio_path: str,
    target_sr: int = DEFAULT_TARGET_SR,
) -> Tuple[np.ndarray, int, MetaData]:
    """
    Stage A: load audio, convert to mono, resample, normalize, and emit metadata.

    Returns a tuple of (waveform, sample_rate, MetaData).
    """

    audio_path = str(Path(audio_path))
    y, original_sr = _load_audio(audio_path)

    if y.size == 0:
        raise ValueError("Loaded audio is empty")

    y_mono = _to_mono(y)
    y_trimmed = _trim_silence(y_mono, top_db=40.0)

    if y_trimmed.size == 0:
        raise ValueError("Audio contains only silence after trimming")

    duration_after_trim = float(len(y_trimmed) / original_sr)
    if duration_after_trim < 0.5:
        raise ValueError("Audio duration after trimming is below 0.5 seconds")

    y_resampled, sr_out = _resample(y_trimmed, original_sr, target_sr)
    y_norm, loudness = _normalize_loudness(y_resampled, sr_out)

    rms_db, peak_level = _validate_signal(y_norm)

    duration_sec = float(len(y_norm) / sr_out)

    meta = MetaData(
        original_sr=original_sr,
        target_sr=sr_out,
        sample_rate=sr_out,
        duration_sec=duration_sec,
        hop_length=DEFAULT_HOP_LENGTH,
        time_signature="4/4",
        tempo_bpm=None,
        detected_key=None,
        lufs=loudness,
        signal_rms_db=rms_db,
        peak_level=peak_level,
        preprocessed_audio=y_norm.astype(np.float32),
    )

    return y_norm.astype(np.float32), sr_out, meta
