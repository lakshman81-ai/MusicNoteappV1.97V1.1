from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from backend.config_manager import get_config
from .separation import SeparationResult, run_htdemucs
from .models import MetaData


def _stage_a_params() -> dict:
    cfg = get_config()
    return {
        "target_sr": int(cfg.get("sample_rate", 44100)),
        "fallback_sr": int(cfg.get("fallback_sample_rate", 22050)),
        "hop_length": int(cfg.get("hop_length", 512)),
        "target_lufs": float(cfg.get("target_lufs", -14.0)),
        "silence_threshold_dbfs": float(cfg.get("silence_threshold_dbfs", 40.0)),
        "min_duration_sec": float(cfg.get("min_duration_sec", 0.12)),
    }


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


def _trim_silence(y: np.ndarray, top_db: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Trim leading and trailing silence using an energy threshold.

    Returns the trimmed audio and the sample indices (start, end) used for trimming.
    """

    try:
        non_silent_indices = librosa.effects.split(y, top_db=top_db)
    except Exception:
        return y.astype(np.float32), (0, len(y))

    if non_silent_indices.size == 0:
        return np.array([], dtype=np.float32), (0, 0)

    start, end = non_silent_indices[0][0], non_silent_indices[-1][1]
    return y[start:end].astype(np.float32), (int(start), int(end))


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


def _normalize_loudness(y: np.ndarray, sr: int, target_lufs: float) -> Tuple[np.ndarray, float | None]:
    """Normalize loudness to a target LUFS while enforcing peak ∈ [−1, 1]."""

    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        y_norm = pyln.normalize.loudness(y, loudness, target_lufs)
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
    target_sr: int | None = None,
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
    params = _stage_a_params()
    target_sr = target_sr or params["target_sr"]
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

    separation_cfg: Dict[str, object] = get_config().get("separation", {})
    separation_enabled = bool(separation_cfg.get("enabled", False))
    separation_target_lufs = float(separation_cfg.get("target_lufs", params["target_lufs"]))
    normalize_stems = bool(separation_cfg.get("normalize", True))
    separation_result: SeparationResult | None = None
    if separation_enabled:
        separation_result = run_htdemucs(
            y,
            original_sr,
            model_name=str(separation_cfg.get("model", "htdemucs")),
            device_preference=str(separation_cfg.get("device", "auto")),
        )

    if y.ndim > 1:
        y = y.astype(np.float32)
    processing_mode = "mono"
    if stereo_mode and y.ndim > 1:
        y, processing_mode = _mid_side_select(y)
    else:
        y = _to_mono(y)

    y = y - float(np.mean(y))

    y_trimmed, trim_bounds = _trim_silence(y, top_db=params["silence_threshold_dbfs"])

    if separation_result and separation_result.stems:
        start_idx, end_idx = trim_bounds
        start_time = start_idx / float(original_sr) if original_sr > 0 else 0.0
        end_time = end_idx / float(original_sr) if original_sr > 0 else 0.0
        trimmed_stems: Dict[str, np.ndarray] = {}
        for name, stem_audio in separation_result.stems.items():
            stem_sr = separation_result.sample_rate
            stem_start = int(round(start_time * stem_sr))
            stem_end = int(round(end_time * stem_sr))
            trimmed = stem_audio[stem_start:stem_end]
            trimmed_stems[name] = _to_mono(trimmed)
        separation_result.stems = trimmed_stems

    if y_trimmed.size == 0:
        raise ValueError("Audio contains only silence after trimming")

    duration_after_trim = float(len(y_trimmed) / original_sr)
    if duration_after_trim < params["min_duration_sec"]:
        raise ValueError("Audio duration after trimming is below minimum duration")

    y_resampled, sr_out = _resample(
        y_trimmed, original_sr, target_sr, fallback_sr=params["fallback_sr"]
    )

    stem_tracks: Dict[str, np.ndarray] | None = None
    stem_lufs: Dict[str, float] | None = None
    if separation_result and separation_result.stems:
        stem_tracks = {}
        stem_lufs = {}
        for name, stem_audio in separation_result.stems.items():
            try:
                stem_resampled, _ = _resample(stem_audio, separation_result.sample_rate, sr_out)
            except Exception:
                stem_resampled = stem_audio.astype(np.float32)
            if normalize_stems:
                stem_norm, stem_loudness = _normalize_loudness(
                    stem_resampled, sr_out, separation_target_lufs
                )
            else:
                stem_norm = stem_resampled.astype(np.float32)
                stem_loudness = None
            stem_tracks[name] = stem_norm.astype(np.float32)
            if stem_loudness is not None:
                stem_lufs[name] = stem_loudness

    try:
        tuning_offset = float(librosa.estimate_tuning(y=y_resampled, sr=sr_out) * 100.0)
    except Exception:
        tuning_offset = 0.0
    if tuning_offset != 0.0:
        try:
            y_resampled = librosa.effects.pitch_shift(y_resampled, sr=sr_out, n_steps=-tuning_offset / 100.0)
        except Exception:
            pass

    y_norm, loudness = _normalize_loudness(y_resampled, sr_out, params["target_lufs"])

    rms_db, peak_level = _validate_signal(y_norm)

    duration_sec = float(len(y_norm) / sr_out)
    window_size = _adaptive_window(y_norm, sr_out)

    meta = MetaData(
        original_sr=original_sr,
        target_sr=sr_out,
        sample_rate=sr_out,
        duration_sec=duration_sec,
        hop_length=params["hop_length"],
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

    if stem_tracks:
        meta.stems = stem_tracks
        meta.stems_sr = sr_out
        meta.stem_lufs = stem_lufs
        meta.separation_model = separation_result.model_name if separation_result else None
        meta.separation_device = separation_result.device if separation_result else None

    return y_norm.astype(np.float32), sr_out, meta
