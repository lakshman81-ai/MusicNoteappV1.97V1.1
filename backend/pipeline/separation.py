from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SeparationResult:
    stems: Dict[str, np.ndarray]
    sample_rate: int
    model_name: str
    device: str


def _select_device(preference: str) -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    pref = (preference or "auto").lower()
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if pref in {"cpu"}:
        return "cpu"
    if pref in {"cuda", "gpu"} and has_cuda:
        return "cuda"
    if pref == "mps" and has_mps:
        return "mps"

    if pref == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
    return "cuda" if has_cuda else "mps" if has_mps else "cpu"


def run_htdemucs(
    audio: np.ndarray,
    sample_rate: int,
    *,
    model_name: str = "htdemucs",
    device_preference: str = "auto",
) -> Optional[SeparationResult]:
    """Run HTDemucs separation on an in-memory waveform.

    The input ``audio`` is expected to be shaped as (samples,) or (samples, channels).
    Returns None if Demucs or its dependencies are unavailable.
    """

    try:
        import torch
        from demucs import pretrained
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
    except Exception:
        return None

    if audio.ndim == 1:
        audio_tensor = torch.tensor(audio[None, :], dtype=torch.float32)
    else:
        # Convert to (channels, time)
        audio_tensor = torch.tensor(audio.T, dtype=torch.float32)

    device = _select_device(device_preference)
    model = pretrained.get_model(model_name)
    model.to(device)
    model.eval()

    # Convert to the model's expected sample rate and channel count
    audio_tensor = convert_audio(
        audio_tensor,
        sample_rate,
        model.samplerate,
        model.audio_channels,
    )

    with torch.no_grad():
        sources = apply_model(model, audio_tensor[None], device=device, progress=False, num_workers=0)[0]

    stems: Dict[str, np.ndarray] = {}
    for name, stem_audio in zip(model.sources, sources):
        # stem_audio shape: (channels, time)
        stems[name] = np.mean(stem_audio.cpu().numpy(), axis=0).astype(np.float32)

    return SeparationResult(stems=stems, sample_rate=int(model.samplerate), model_name=model_name, device=device)
