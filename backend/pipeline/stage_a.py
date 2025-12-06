from __future__ import annotations

from pathlib import Path
from typing import Tuple
import math

import numpy as np
import librosa
from scipy import signal as sps

from .models import MetaData

# Internal target sample rate
TARGET_SR = 22050


def _resample_to_target(y: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    """
    Resample using scipy.signal.resample_poly to avoid the resampy dependency.
    """
    if sr == target_sr:
        return y, sr

    g = math.gcd(int(sr), int(target_sr))
    up = target_sr // g
    down = sr // g
    y_resampled = sps.resample_poly(y, up, down)
    return y_resampled.astype(np.float32), target_sr


def load_and_preprocess(
    audio_path: str | Path,
    stereo_mode: str | None = None,
    start_offset: float = 0.0,
    **kwargs,
) -> Tuple[np.ndarray, int, MetaData]:
    """
    Stage A: Load audio, convert to mono, normalize, resample, and create MetaData.

    Args (for backward compatibility):
        stereo_mode: ignored; we always convert to mono.
        start_offset: seconds to skip at the start of the file.
        **kwargs: ignored safely.

    Returns:
        y    : mono waveform (float32)
        sr   : sample rate (int)
        meta : MetaData instance populated with basic info
    """
    audio_path = str(audio_path)

    # 1. Load raw audio at native sample rate, mono
    y, sr = librosa.load(audio_path, sr=None, mono=True, offset=float(start_offset))

    if y.size == 0:
        raise ValueError("Empty audio file")

    # 2. Normalize amplitude
    max_val = float(np.max(np.abs(y)))
    if max_val > 0:
        y = y / max_val

    # 3. Resample to TARGET_SR
    y, sr = _resample_to_target(y, sr, TARGET_SR)

    # 4. Build MetaData with safe defaults
    meta = MetaData()

    if hasattr(meta, "sample_rate"):
        meta.sample_rate = sr

    if hasattr(meta, "hop_length"):
        # 512 at 22.05 kHz ≈ 23 ms
        meta.hop_length = getattr(meta, "hop_length", 512) or 512

    if hasattr(meta, "tempo_bpm"):
        meta.tempo_bpm = getattr(meta, "tempo_bpm", 120.0) or 120.0

    if hasattr(meta, "time_signature"):
        meta.time_signature = getattr(meta, "time_signature", "4/4") or "4/4"

    if hasattr(meta, "detected_key"):
        meta.detected_key = getattr(meta, "detected_key", None)

    return y.astype(np.float32), sr, meta
