from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .models import MetaData


# Stage A constants aligned to the deterministic specification
DEFAULT_TARGET_SR = 44100
FALLBACK_SR = 22050
DEFAULT_HOP_LENGTH = 512
TARGET_LUFS = -14.0
SILENCE_THRESHOLD_DBFS = 40.0
MIN_DURATION_SEC = 0.12


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


def _normalize_channel_orientation(y: np.ndarray) -> tuple[np.ndarray, str, bool]:
    """Ensure multi-channel audio is shaped as (samples, channels).

    Some loaders (e.g., ``librosa.load`` with ``mono=False``) return arrays shaped
    as ``(channels, samples)``. This function transposes those arrays so that
    downstream stereo processing receives the canonical ``(samples, channels)``
    layout.
    """

    if y.ndim != 2:
        return y.astype(np.float32), "mono", False

    channels_first = y.shape[0] < y.shape[1] and y.shape[0] <= 8
    if channels_first:
        return y.T.astype(np.float32), "channels_first", True
    return y.astype(np.float32), "samples_first", False


def _trim_silence(y: np.ndarray, top_db: float = SILENCE_THRESHOLD_DBFS) -> np.ndarray:
    """Trim leading and trailing silence using an energy threshold."""

    try:
        non_silent_indices = librosa.effects.split(y, top_db=top_db)
    except Exception:
        return y.astype(np.float32)

    if non_silent_indices.size == 0:
        return np.array([], dtype=np.float32)

    start, end = non_silent_indices[0][0], non_silent_indices[-1][1]
    return y[start:end].astype(np.float32)


def _resample(y: np.ndarray, sr: int, target_sr: int, fallback_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Resample waveform to the target sample rate with an optional fallback."""

    if sr == target_sr:
        return y.astype(np.float32), sr
    try:
        y_resampled = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        return y_resampled.astype(np.float32), target_sr
    except Exception:
        if fallback_sr and sr != fallback_sr:
            y_fallback = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=fallback_sr)
            return y_fallback.astype(np.float32), fallback_sr
        raise


def _normalize_loudness(y: np.ndarray, sr: int) -> Tuple[np.ndarray, float | None]:
    """Normalize loudness to TARGET_LUFS while enforcing peak ∈ [−1, 1]."""

    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        y_norm = pyln.normalize.loudness(y, loudness, TARGET_LUFS)
    except Exception:
        y_norm = y.astype(np.float32)
        loudness = None

    peak = float(np.max(np.abs(y_norm)))
    if peak > 0:
        y_norm = np.clip(y_norm / peak, -1.0, 1.0)
    return y_norm.astype(np.float32), None if loudness is None else float(loudness)


def _validate_signal(y: np.ndarray, rms_floor_db: float = -50.0, peak_floor: float = 1e-4) -> Tuple[float, float]:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    rms_db = 20.0 * np.log10(max(rms, 1e-12))
    if peak < peak_floor or rms_db < rms_floor_db:
        raise ValueError(
            f"Audio lacks usable signal after normalization (peak={peak:.5f}, rms_db={rms_db:.1f})"
        )
    return rms_db, peak


def _mid_side_select(y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Select Mid or Side channel based on energy to mitigate phase cancellation."""

    if y.ndim == 1 or y.shape[1] < 2:
        return _to_mono(y), "mono"

    left = y[:, 0]
    right = y[:, 1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    def _energy(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    mid_energy = _energy(mid)
    side_energy = _energy(side)
    if side_energy > mid_energy * 1.05:
        return side.astype(np.float32), "stereo-side"
    return mid.astype(np.float32), "stereo-mid"


def _adaptive_window(y: np.ndarray, sr: int) -> int:
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = float(np.nanmean(centroid)) if centroid.size else 0.0
    return 1024 if avg_centroid >= 2000.0 else 4096


def load_and_preprocess(
    audio_path: str,
    target_sr: int = DEFAULT_TARGET_SR,
    stereo_mode: bool = False,
    start_offset: float = 0.0,
    max_duration: float | None = None,
) -> Tuple[np.ndarray, int, MetaData]:
    """
    Stage A: load audio, remove DC offset, normalize loudness, optionally handle
    stereo mid/side selection, and emit metadata.

    Returns a tuple of (waveform, sample_rate, MetaData).
    """

    audio_path = str(Path(audio_path))
    y, original_sr = _load_audio(audio_path)

    input_shape = tuple(y.shape)
    y, orientation, transposed = _normalize_channel_orientation(y)
    normalized_shape = tuple(y.shape)

    if y.size == 0:
        raise ValueError("Loaded audio is empty")

    if start_offset > 0.0:
        start_sample = int(start_offset * original_sr)
        y = y[start_sample:]
    if max_duration is not None and max_duration > 0:
        end_sample = int(max_duration * original_sr)
        y = y[:end_sample]

    if y.ndim > 1:
        y = y.astype(np.float32)
    processing_mode = "mono"
    if stereo_mode and y.ndim > 1:
        y, processing_mode = _mid_side_select(y)
    else:
        y = _to_mono(y)

    y = y - float(np.mean(y))

    y_trimmed = _trim_silence(y, top_db=SILENCE_THRESHOLD_DBFS)

    if y_trimmed.size == 0:
        raise ValueError("Audio contains only silence after trimming")

    duration_after_trim = float(len(y_trimmed) / original_sr)
    if duration_after_trim < MIN_DURATION_SEC:
        raise ValueError("Audio duration after trimming is below minimum duration")

    y_resampled, sr_out = _resample(y_trimmed, original_sr, target_sr, fallback_sr=FALLBACK_SR)

    try:
        tuning_offset = float(librosa.estimate_tuning(y=y_resampled, sr=sr_out) * 100.0)
    except Exception:
        tuning_offset = 0.0
    if tuning_offset != 0.0:
        try:
            y_resampled = librosa.effects.pitch_shift(y_resampled, sr=sr_out, n_steps=-tuning_offset / 100.0)
        except Exception:
            pass

    y_norm, loudness = _normalize_loudness(y_resampled, sr_out)

    rms_db, peak_level = _validate_signal(y_norm)

    duration_sec = float(len(y_norm) / sr_out)
    window_size = _adaptive_window(y_norm, sr_out)

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
    meta.processing_mode = processing_mode
    meta.window_size = window_size
    meta.tuning_offset = tuning_offset
    meta.channel_orientation = orientation if not transposed else "samples_first"
    meta.original_shape = input_shape
    meta.normalized_shape = normalized_shape

    return y_norm.astype(np.float32), sr_out, meta
