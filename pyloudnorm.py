"""Lightweight fallback for pyloudnorm when the dependency is unavailable.

This is not a full implementation; it provides the minimal API used by
``backend.pipeline.stage_a`` for benchmarking workflows in constrained
environments.
"""
from __future__ import annotations

import numpy as np


class Meter:
    def __init__(self, rate: int):
        self.rate = rate

    def integrated_loudness(self, signal: np.ndarray) -> float:
        signal = np.asarray(signal, dtype=float)
        if signal.size == 0:
            return -np.inf
        rms = float(np.sqrt(np.mean(np.square(signal))))
        return 20.0 * np.log10(max(rms, 1e-12))


class normalize:
    @staticmethod
    def loudness(signal: np.ndarray, current_lufs: float, target_lufs: float) -> np.ndarray:
        if not np.isfinite(current_lufs):
            return signal.astype(np.float32)
        gain_db = target_lufs - current_lufs
        gain = 10 ** (gain_db / 20.0)
        return (signal.astype(np.float32) * gain).astype(np.float32)
