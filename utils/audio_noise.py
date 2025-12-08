from __future__ import annotations

import numpy as np


def add_noise_snr(clean: np.ndarray, snr_db: float, *, seed: int | None = None) -> np.ndarray:
    """Add white noise to `clean` at a target SNR level in dB."""

    clean = np.asarray(clean, dtype=float)
    if clean.size == 0:
        return clean

    signal_power = np.mean(clean**2)
    if signal_power <= 0:
        return clean

    rng = np.random.default_rng(seed)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power), size=clean.shape)
    return clean + noise
