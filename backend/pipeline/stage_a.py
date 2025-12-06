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
    y_resampled, sr_out = _resample(y_mono, original_sr, target_sr)
    y_norm, loudness = _normalize_loudness(y_resampled, sr_out)

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
    )

    return y_norm.astype(np.float32), sr_out, meta
