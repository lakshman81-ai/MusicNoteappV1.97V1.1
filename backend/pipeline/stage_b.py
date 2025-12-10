from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

import librosa
import numpy as np
from scipy import linalg, signal
from scipy.ndimage import gaussian_filter1d

from backend.config_manager import get_config
from .models import ChordEvent, FramePitch, MetaData, NoteEvent


def _stage_b_params() -> dict:
    cfg = get_config()
    weights = cfg.get("ensemble_weights", {})
    default_weights: Dict[str, float] = {
        "swift": 0.45,
        "crepe": 0.25,
        "rmvpe": 0.25,
        "yin": 0.15,
        "cqt": 0.10,
        "autocorr": 0.10,
    }
    default_weights.update({k.lower(): v for k, v in weights.items()})
    return {
        "fmin": float(cfg.get("fmin", 65.0)),
        "fmax": float(cfg.get("fmax", 2093.0)),
        "conf_min": float(cfg.get("confidence_floor", 0.10)),
        "frame_length": int(cfg.get("frame_length", 2048)),
        "hop_length": int(cfg.get("hop_length", 512)),
        "median_window": int(cfg.get("median_window", 11)),
        "weights": default_weights,
        "onset_threshold_factor": float(cfg.get("onset_threshold_factor", 0.25)),
        "min_note_frames": int(cfg.get("min_note_frames", 3)),
        "autocorr_split_hz": float(cfg.get("autocorr_split_hz", 1000.0)),
        "autocorr_whitening_order": int(cfg.get("autocorr_whitening_order", 12)),
        "autocorr_whitening_lambda": float(cfg.get("autocorr_whitening_lambda", 0.01)),
        "autocorr_peak_threshold": float(cfg.get("autocorr_peak_threshold", 0.20)),
        "autocorr_octave_compression": float(cfg.get("autocorr_octave_compression", 2.0)),
        "autocorr_octave_suppression": float(cfg.get("autocorr_octave_suppression", 0.50)),
        "swift_peak_threshold": float(cfg.get("swift_peak_threshold", 0.15)),
        "swift_peak_prominence": float(cfg.get("swift_peak_prominence", 0.05)),
        "swift_gaussian_sigma": float(cfg.get("swift_gaussian_sigma", 1.0)),
        "swift_max_peaks": int(cfg.get("swift_max_peaks", 5)),
        "swift_polyphony_cap": int(cfg.get("swift_polyphony_cap", 2)),
        "swift_softmax_min_hz": float(cfg.get("swift_softmax_min_hz", cfg.get("fmin", 65.0))),
        "swift_softmax_max_hz": float(cfg.get("swift_softmax_max_hz", cfg.get("fmax", 2093.0))),
        "swift_num_harmonics": int(cfg.get("swift_num_harmonics", 6)),
        "swift_harmonic_width": int(cfg.get("swift_harmonic_width", 3)),
        "swift_harmonic_attenuation": float(cfg.get("swift_harmonic_attenuation", 0.35)),
        "swift_sacf_threshold": float(cfg.get("swift_sacf_threshold", 0.2)),
        "swift_octave_tolerance_cents": float(cfg.get("swift_octave_tolerance_cents", 15.0)),
    }


def _mix_stems(meta: MetaData, names: List[str]) -> Tuple[np.ndarray, int] | None:
    stems = meta.stems or {}
    tracks = [stems[name] for name in names if name in stems and stems[name] is not None]
    if not tracks:
        return None

    max_len = max(len(track) for track in tracks)
    padded = []
    for track in tracks:
        pad_len = max_len - len(track)
        if pad_len > 0:
            track = np.pad(track, (0, pad_len), mode="constant")
        padded.append(track.astype(np.float32))

    mix = np.mean(np.stack(padded, axis=0), axis=0).astype(np.float32)
    return mix, int(meta.stems_sr or meta.sample_rate)


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.empty_like(values)
    for i in range(len(values)):
        smoothed[i] = np.median(padded[i : i + window])
    return smoothed


def _estimate_tuning_offset(timeline: List[FramePitch]) -> float:
    """Estimate tuning offset in cents from the nearest MIDI bins."""

    cents_diffs: List[float] = []
    weights: List[float] = []
    for frame in timeline:
        if frame.pitch_hz <= 0 or frame.confidence <= 0:
            continue
        midi_val = float(librosa.hz_to_midi(frame.pitch_hz))
        nearest = round(midi_val)
        cents = 1200.0 * np.log2(frame.pitch_hz / float(librosa.midi_to_hz(nearest)))
        cents_diffs.append(float(cents))
        weights.append(float(frame.confidence))

    if not cents_diffs:
        return 0.0

    bins = np.linspace(-50.0, 50.0, 41)  # 2.5-cent bins
    hist, edges = np.histogram(cents_diffs, bins=bins, weights=weights)
    if not np.any(hist):
        return 0.0

    best_idx = int(np.argmax(hist))
    center = 0.5 * (edges[best_idx] + edges[best_idx + 1])
    return float(center)


def _decode_hmm_states(timeline: List[FramePitch], frame_duration: float, min_note_frames: int) -> List[str]:
    """
    Lightweight 3-state HMM (silence/attack/stable) over the pitch timeline.

    Uses confidence, SACF salience, and simple transition priors to encourage
    realistic attack–sustain–release flows while enforcing minimum dwell times.
    """

    states = ["silence", "attack", "stable"]

    log_prior = np.log(np.asarray([0.6, 0.2, 0.2]))
    log_trans = np.log(
        np.array(
            [
                [0.85, 0.10, 0.05],  # silence -> {silence, attack, stable}
                [0.15, 0.65, 0.20],  # attack  -> {silence, attack, stable}
                [0.20, 0.10, 0.70],  # stable  -> {silence, attack, stable}
            ]
        )
    )

    def _emission_log_prob(frame: FramePitch, state: str) -> float:
        voiced = frame.pitch_hz > 0
        conf = float(np.clip(frame.confidence, 0.0, 1.0))
        sal = float(np.clip(frame.salience, 0.0, 1.0))
        base = 1e-6

        if state == "silence":
            likelihood = base + (1.2 - conf) * (1.0 - 0.3 * sal)
            return float(np.log(max(likelihood, base)))

        if state == "attack":
            likelihood = base + conf * 0.55 + sal * 0.35 + (0.15 if voiced else -0.2)
            return float(np.log(max(likelihood, base)))

        # stable
        likelihood = base + conf * 0.75 + sal * 0.4 + (0.2 if voiced else -0.4)
        return float(np.log(max(likelihood, base)))

    n_frames = len(timeline)
    if n_frames == 0:
        return []

    dp = np.full((len(states), n_frames), -np.inf)
    backptr = np.full((len(states), n_frames), -1, dtype=int)

    # initialization
    first_emissions = [_emission_log_prob(timeline[0], s) for s in states]
    for idx, em in enumerate(first_emissions):
        dp[idx, 0] = log_prior[idx] + em

    # Viterbi forward pass
    for t in range(1, n_frames):
        frame = timeline[t]
        emissions = [_emission_log_prob(frame, s) for s in states]
        for s_idx, emission in enumerate(emissions):
            prev_scores = dp[:, t - 1] + log_trans[:, s_idx]
            best_prev = int(np.argmax(prev_scores))
            dp[s_idx, t] = prev_scores[best_prev] + emission
            backptr[s_idx, t] = best_prev

    # backtrack
    state_path = [0] * n_frames
    state_path[-1] = int(np.argmax(dp[:, -1]))
    for t in range(n_frames - 2, -1, -1):
        state_path[t] = int(backptr[state_path[t + 1], t + 1])

    decoded = [states[idx] for idx in state_path]

    # Enforce minimum dwell time by merging short segments into neighbors
    min_frames = max(1, min_note_frames)
    start = 0
    while start < n_frames:
        end = start
        while end + 1 < n_frames and decoded[end + 1] == decoded[start]:
            end += 1
        run_length = end - start + 1
        if run_length < min_frames and start > 0 and end < n_frames - 1:
            prev_state = decoded[start - 1]
            next_state = decoded[end + 1]
            decoded[start : end + 1] = [prev_state if prev_state == next_state else next_state] * run_length
        start = end + 1

    return decoded


def _detector_yin(
    y: np.ndarray, sr: int, fmin: float, fmax: float, frame_length: int, hop_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    yin_pitch = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length)
    conf = np.where(np.isfinite(yin_pitch), 0.6, 0.0)
    return yin_pitch.astype(float), conf.astype(float)


def _detector_cqt(y: np.ndarray, sr: int, fmin: float, fmax: float, hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = int(np.ceil(12 * np.log2(fmax / fmin))) * 3
    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=36,
        pad_mode="reflect",
    )
    mags = np.abs(cqt)
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=36)
    pitches = np.zeros(mags.shape[1], dtype=float)
    conf = np.zeros_like(pitches)
    for idx in range(mags.shape[1]):
        frame = mags[:, idx]
        if not np.any(np.isfinite(frame)):
            pitches[idx] = np.nan
            continue
        peak_idx = int(np.argmax(frame))
        pitches[idx] = float(freqs[peak_idx])
        frame_db = librosa.amplitude_to_db(frame + 1e-6, ref=np.max)
        conf[idx] = float(np.clip((np.max(frame_db) - np.median(frame_db)) / 40.0, 0.0, 1.0))
    return pitches, conf


def _whiten_frame(frame: np.ndarray, order: int, reg: float) -> np.ndarray:
    frame = frame - float(np.mean(frame))
    if order <= 0 or not np.any(frame):
        return frame

    order = min(order, len(frame) - 1)
    if order <= 0:
        return frame

    ac = librosa.autocorrelate(frame, max_size=order + 1)
    if ac.size <= order:
        return frame

    toeplitz = linalg.toeplitz(ac[:-1]) + reg * np.eye(order)
    rhs = -ac[1:]
    try:
        coeffs = linalg.solve(toeplitz, rhs, assume_a="pos")
    except linalg.LinAlgError:
        return frame

    a = np.concatenate(([1.0], coeffs))
    return signal.lfilter(a, [1.0], frame)


def _design_band_filters(sr: int, split_hz: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    nyquist = sr / 2.0
    split = np.clip(split_hz, 10.0, nyquist - 100.0)
    norm_split = split / nyquist
    low_b, low_a = signal.butter(4, norm_split, btype="low")
    high_b, high_a = signal.butter(4, norm_split, btype="high")
    return (low_b, low_a), (high_b, high_a)


def _apply_octave_pruning(acf: np.ndarray, compression: float, suppression: float) -> np.ndarray:
    if compression <= 1.0 or suppression <= 0.0:
        return acf

    compressed_len = max(1, int(np.ceil(len(acf) / compression)))
    compressed = signal.resample(acf, compressed_len)
    compressed_time = np.arange(compressed_len) * compression
    compressed_full = np.interp(
        np.arange(len(acf)), compressed_time, compressed, left=compressed[0], right=0.0
    )
    pruned = acf - suppression * compressed_full
    return np.maximum(pruned, 0.0)


def _detector_autocorr(
    y: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    frame_length: int,
    hop_length: int,
    split_hz: float,
    whitening_order: int,
    whitening_lambda: float,
    peak_threshold: float,
    octave_compression: float,
    octave_suppression: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    pitches = np.full(frames.shape[0], np.nan, dtype=float)
    conf = np.zeros_like(pitches)
    salience = np.zeros_like(pitches)

    (low_b, low_a), (high_b, high_a) = _design_band_filters(sr, split_hz)

    for i, frame in enumerate(frames):
        whitened = _whiten_frame(frame.astype(float), whitening_order, whitening_lambda)
        if not np.any(np.isfinite(whitened)):
            continue

        low_band = signal.lfilter(low_b, low_a, whitened)
        high_band = np.maximum(signal.lfilter(high_b, high_a, whitened), 0.0)

        ac_low = librosa.autocorrelate(low_band)
        ac_high = librosa.autocorrelate(high_band)

        max_len = min(len(ac_low), len(ac_high))
        if max_len == 0:
            continue

        summary_acf = ac_low[:max_len] + ac_high[:max_len]
        if not np.any(np.isfinite(summary_acf)):
            continue

        summary_acf = summary_acf / (np.max(np.abs(summary_acf)) + 1e-9)
        summary_acf = _apply_octave_pruning(summary_acf, octave_compression, octave_suppression)

        summary_acf[:min_lag] = 0.0
        summary_acf[max_lag:] = 0.0

        peak_idx = int(np.argmax(summary_acf))
        peak_val = float(summary_acf[peak_idx]) if peak_idx > 0 else 0.0
        salience[i] = peak_val

        if peak_idx <= 0 or peak_val < peak_threshold:
            continue

        pitches[i] = sr / float(peak_idx)
        conf[i] = float(np.clip(peak_val, 0.0, 1.0))

    return pitches, conf, salience


def _detector_swift(
    y: np.ndarray, sr: int, hop_length: int, frame_length: int, params: dict
) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[float, float]]]] | None:
    if importlib.util.find_spec("swift") is None and importlib.util.find_spec("swiftf0") is None:
        return None
    try:
        swift = importlib.import_module("swift").SwiftF0
    except Exception:
        try:
            swift = importlib.import_module("swiftf0").SwiftF0
        except Exception:
            return None

    def _run_swift(model_input: np.ndarray, model_sr: int, model_hop: int):
        model = swift(sample_rate=model_sr)
        result = model.infer(model_input, hop_size=model_hop)
        pitches = np.asarray(result.get("f0_hz", []), dtype=float)
        conf = np.asarray(result.get("confidence", np.ones_like(pitches)), dtype=float)
        softmax = result.get("softmax") or result.get("probabilities") or result.get("probs") or None
        softmax_arr = np.asarray(softmax, dtype=float) if softmax is not None else None
        return pitches, conf, softmax_arr

    cfg = get_config()
    target_sr = 16000
    pipeline_sr = int(cfg.get("sample_rate", sr))
    pipeline_at_target_sr = pipeline_sr == target_sr
    already_at_target_sr = sr == target_sr or (pipeline_at_target_sr and sr == pipeline_sr)

    if not already_at_target_sr:
        y_swift = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        swift_sr = target_sr
        swift_hop = int(round(hop_length * target_sr / sr))
    else:
        y_swift = y
        swift_sr = sr
        swift_hop = hop_length

    pitches, conf, softmax = _run_swift(y_swift, swift_sr, swift_hop)
    if pitches.size == 0:
        return None

    freq_axis = _swift_frequency_axis(params["swift_max_peaks"], params["swift_softmax_min_hz"], params["swift_softmax_max_hz"])
    candidates: List[List[Tuple[float, float]]] = []
    if softmax is not None and softmax.ndim > 1:
        freq_axis = _swift_frequency_axis(softmax.shape[1], params["swift_softmax_min_hz"], params["swift_softmax_max_hz"])
        candidates = _extract_swift_candidates_from_softmax(
            softmax,
            freq_axis,
            params["swift_peak_threshold"],
            params["swift_peak_prominence"],
            params["swift_gaussian_sigma"],
            params["swift_max_peaks"],
        )

    # Fall back to the model pitch as a candidate when softmax is missing.
    if not candidates:
        candidates = [[(float(f), float(c))] if np.isfinite(f) and f > 0 else [] for f, c in zip(pitches, conf)]

    validated = _validate_swift_with_sacf(
        y_swift, swift_sr, swift_hop, candidates, params["swift_sacf_threshold"], frame_length
    )
    pruned_validated = [_prune_harmonic_ghosts(frame, params["swift_octave_tolerance_cents"]) for frame in validated]

    primary_freqs = np.full(len(pruned_validated), np.nan, dtype=float)
    primary_conf = np.zeros(len(pruned_validated), dtype=float)
    for idx, frame in enumerate(pruned_validated):
        if frame:
            primary_freqs[idx] = frame[0][0]
            primary_conf[idx] = frame[0][1]

    # Iterative spectral subtraction to reveal quieter notes.
    residual_candidates: List[List[Tuple[float, float]]] = []
    if params["swift_polyphony_cap"] > 1:
        residual_signal = _spectral_subtract_harmonics(
            y_swift,
            swift_sr,
            swift_hop,
            frame_length,
            primary_freqs,
            params["swift_num_harmonics"],
            params["swift_harmonic_width"],
            params["swift_harmonic_attenuation"],
        )
        residual_pitches, residual_conf, residual_softmax = _run_swift(residual_signal, swift_sr, swift_hop)
        freq_axis_residual = freq_axis
        if residual_softmax is not None and residual_softmax.ndim > 1:
            if residual_softmax.shape[1] != freq_axis.shape[0]:
                freq_axis_residual = _swift_frequency_axis(
                    residual_softmax.shape[1], params["swift_softmax_min_hz"], params["swift_softmax_max_hz"]
                )
            residual_candidates = _extract_swift_candidates_from_softmax(
                residual_softmax,
                freq_axis_residual,
                params["swift_peak_threshold"],
                params["swift_peak_prominence"],
                params["swift_gaussian_sigma"],
                params["swift_max_peaks"],
            )
        if not residual_candidates:
            residual_candidates = [
                [(float(f), float(c))] if np.isfinite(f) and f > 0 else [] for f, c in zip(residual_pitches, residual_conf)
            ]
        residual_validated = _validate_swift_with_sacf(
            residual_signal,
            swift_sr,
            swift_hop,
            residual_candidates,
            params["swift_sacf_threshold"],
            frame_length,
        )
    else:
        residual_validated = []

    merged_candidates: List[List[Tuple[float, float]]] = []
    for idx in range(max(len(pruned_validated), len(residual_validated))):
        frame: List[Tuple[float, float]] = []
        if idx < len(pruned_validated):
            frame.extend(pruned_validated[idx])
        if idx < len(residual_validated):
            frame.extend(residual_validated[idx])
        frame = _prune_harmonic_ghosts(frame, params["swift_octave_tolerance_cents"])
        frame.sort(key=lambda pair: pair[1], reverse=True)
        merged_candidates.append(frame[: params["swift_polyphony_cap"]])

    merged_primary = np.full(len(merged_candidates), np.nan, dtype=float)
    merged_conf = np.zeros(len(merged_candidates), dtype=float)
    for idx, frame in enumerate(merged_candidates):
        if frame:
            merged_primary[idx] = frame[0][0]
            merged_conf[idx] = frame[0][1]

    swift_duration = len(y_swift) / float(swift_sr) if swift_sr > 0 else 0.0
    original_duration = len(y) / float(sr) if sr > 0 else 0.0
    time_scale = (original_duration / swift_duration) if swift_duration > 0 else 1.0
    swift_times = np.arange(len(merged_primary)) * (swift_hop / float(swift_sr)) * time_scale
    target_frames = int(np.ceil(len(y) / hop_length))
    target_times = np.arange(target_frames) * (hop_length / float(sr))

    aligned_pitches = np.interp(target_times, swift_times, merged_primary, left=np.nan, right=np.nan)
    aligned_conf = np.interp(target_times, swift_times, merged_conf, left=0.0, right=0.0)
    aligned_candidates = _align_candidates_to_targets(swift_times, target_times, merged_candidates)

    return aligned_pitches, aligned_conf, aligned_candidates


def _detector_crepe(y: np.ndarray, sr: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray] | None:
    if importlib.util.find_spec("crepe") is None:
        return None
    crepe = importlib.import_module("crepe")
    step_ms = hop_length * 1000.0 / sr
    if sr != 16000:
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr_used = 16000
    else:
        y_resampled = y
        sr_used = sr
    time, frequency, confidence, _ = crepe.predict(y_resampled, sr_used, step_size=step_ms, viterbi=True)
    return frequency.astype(float), confidence.astype(float)


def _aggregate(
    detector_outputs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    conf_min: float,
    weights: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_len = min(len(v[0]) for v in detector_outputs.values()) if detector_outputs else 0
    pitches = np.full(reference_len, np.nan, dtype=float)
    conf = np.zeros(reference_len, dtype=float)
    unstable = np.zeros(reference_len, dtype=bool)

    for idx in range(reference_len):
        frame_candidates = []
        frame_weights = []
        frame_conf = []
        for name, (freqs, probs) in detector_outputs.items():
            freq = freqs[idx]
            prob = float(probs[idx]) if idx < len(probs) else 0.0
            if not np.isfinite(freq) or freq <= 0:
                continue
            frame_candidates.append(freq)
            frame_conf.append(prob)
            frame_weights.append(weights.get(name, 0.1))

        if not frame_candidates:
            continue

        # Swift override rule
        if "swift" in detector_outputs:
            swift_freq = detector_outputs["swift"][0][idx]
            swift_conf = float(detector_outputs["swift"][1][idx]) if idx < len(detector_outputs["swift"][1]) else 0.0
            if np.isfinite(swift_freq) and swift_freq > 0 and swift_conf >= 0.50:
                pitches[idx] = swift_freq
                conf[idx] = max(swift_conf, conf_min)
                unstable[idx] = False
                continue

        cents = np.asarray([1200.0 * np.log2(f / frame_candidates[0]) for f in frame_candidates if f > 0])
        if len(cents) >= 2 and np.any(np.abs(cents - cents[0]) > 120.0):
            best_idx = int(np.argmax(frame_conf))
            pitches[idx] = frame_candidates[best_idx]
            conf[idx] = max(frame_conf[best_idx], conf_min)
            unstable[idx] = True
            continue

        weights = np.asarray(frame_weights, dtype=float)
        weights = weights / np.sum(weights)
        pitches[idx] = float(np.sum(np.asarray(frame_candidates) * weights))
        conf[idx] = float(np.mean(frame_conf))
    return pitches, conf, unstable


def _detect_chords_from_chroma(y: np.ndarray, sr: int, hop_length: int) -> List[ChordEvent]:
    chords: List[ChordEvent] = []
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    except Exception:
        return chords

    frames_per_second = max(1, int(round(sr / hop_length)))
    major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

    for start_frame in range(0, chroma.shape[1], frames_per_second):
        end_frame = min(chroma.shape[1], start_frame + frames_per_second)
        window = chroma[:, start_frame:end_frame]
        if window.size == 0:
            continue
        profile = np.mean(window, axis=1)
        best_score = -np.inf
        best_root = None
        best_quality = None
        for root in range(12):
            rolled_major = np.roll(major_template, root)
            rolled_minor = np.roll(minor_template, root)
            score_major = float(np.dot(profile, rolled_major))
            score_minor = float(np.dot(profile, rolled_minor))
            if score_major > best_score:
                best_score = score_major
                best_root = librosa.midi_to_note(60 + root)[0:-1]
                best_quality = "maj"
            if score_minor > best_score:
                best_score = score_minor
                best_root = librosa.midi_to_note(60 + root)[0:-1]
                best_quality = "min"
        if best_root is None or best_quality is None:
            continue
        time_sec = float(librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length))
        beat_val = float(time_sec * 120.0 / 60.0)
        symbol = f"{best_root}{'' if best_quality == 'maj' else 'm'}"
        chords.append(ChordEvent(time=time_sec, beat=beat_val, symbol=symbol, root=best_root, quality=best_quality))
    return chords


def _map_velocity(rms_value: float, rms_min: float, rms_max: float) -> float:
    rms_db = 20.0 * np.log10(max(rms_value, 1e-12))
    v = 20 + 85 * np.clip((rms_db - rms_min) / (rms_max - rms_min), 0.0, 1.0)
    return float(v)


def extract_features(
    y: np.ndarray,
    sr: int,
    meta: MetaData,
    use_crepe: bool = False,
    velocity_humanization: float | None = None,
    velocity_seed: int | None = None,
) -> Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent]]:
    """Stage B: ensemble pitch detection with HMM-based segmentation."""
    params = _stage_b_params()
    hop_length = params["hop_length"]
    harmonic_source = _mix_stems(meta, ["vocals", "bass", "other"])
    base_y, base_sr = harmonic_source if harmonic_source else (y, sr)
    swift_source = _mix_stems(meta, ["vocals", "bass"])
    poly_source = _mix_stems(meta, ["other"])
    swift_y, swift_sr = swift_source if swift_source else (base_y, base_sr)
    poly_y, poly_sr = poly_source if poly_source else (base_y, base_sr)
    detector_outputs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    yin_pitch, yin_conf = _detector_yin(
        base_y, base_sr, params["fmin"], params["fmax"], params["frame_length"], hop_length
    )
    detector_outputs["yin"] = (yin_pitch, yin_conf)

    cqt_pitch, cqt_conf = _detector_cqt(poly_y, poly_sr, params["fmin"], params["fmax"], hop_length)
    detector_outputs["cqt"] = (cqt_pitch, cqt_conf)

    autocorr_pitch, autocorr_conf, autocorr_salience = _detector_autocorr(
        poly_y,
        poly_sr,
        params["fmin"],
        params["fmax"],
        params["frame_length"],
        hop_length,
        params["autocorr_split_hz"],
        params["autocorr_whitening_order"],
        params["autocorr_whitening_lambda"],
        params["autocorr_peak_threshold"],
        params["autocorr_octave_compression"],
        params["autocorr_octave_suppression"],
    )
    detector_outputs["autocorr"] = (autocorr_pitch, autocorr_conf)

    swift_candidates: List[List[Tuple[float, float]]] = []
    swift_output = _detector_swift(swift_y, swift_sr, hop_length, params["frame_length"], params)
    if swift_output is not None:
        aligned_pitches, aligned_conf, swift_candidates = swift_output
        detector_outputs["swift"] = (aligned_pitches, aligned_conf)

    crepe_output = _detector_crepe(base_y, base_sr, hop_length) if use_crepe else None
    if crepe_output is not None:
        detector_outputs["crepe"] = crepe_output

    pitches, conf, unstable = _aggregate(detector_outputs, params["conf_min"], params["weights"])
    conf = np.clip(conf, 0.0, 1.0)
    conf = np.maximum(conf, params["conf_min"])
    times = librosa.times_like(pitches, sr=base_sr, hop_length=hop_length)

    # Median smoothing respecting NaNs
    pitch_smooth = pitches.copy()
    for idx in range(len(pitches)):
        start = max(0, idx - params["median_window"] // 2)
        end = min(len(pitches), idx + params["median_window"] // 2 + 1)
        window_vals = pitches[start:end]
        window_vals = window_vals[np.isfinite(window_vals)]
        if window_vals.size:
            pitch_smooth[idx] = float(np.median(window_vals))

    rms = librosa.feature.rms(y=base_y, frame_length=params["frame_length"], hop_length=hop_length)[0]
    if rms.size < len(times):
        rms = np.pad(rms, (0, len(times) - rms.size), mode="edge")
    rms_floor = -40.0
    rms_ceiling = -4.0

    timeline: List[FramePitch] = []
    for idx, (t, f, c, unstable_flag) in enumerate(zip(times, pitch_smooth, conf, unstable)):
        midi = None
        if np.isfinite(f) and f > 0:
            midi_val = float(librosa.hz_to_midi(f))
            midi = int(round(midi_val)) if np.isfinite(midi_val) else None
        swift_salience = swift_candidates[idx][0][1] if idx < len(swift_candidates) and swift_candidates[idx] else 0.0
        autocorr_val = float(autocorr_salience[idx]) if idx < len(autocorr_salience) else 0.0
        salience = max(swift_salience, autocorr_val)
        timeline.append(
            FramePitch(
                time=float(t),
                pitch_hz=float(f) if np.isfinite(f) else 0.0,
                midi=midi,
                confidence=float(c),
                salience=salience,
            )
        )
        if unstable_flag:
            timeline[-1].confidence = max(float(c), params["conf_min"])

    # Stage C rules: HMM-driven segmentation with hysteresis and tuning-aware quantization
    frame_duration = float(hop_length / sr)
    silence_frames_required = int(np.ceil(0.12 / frame_duration))
    hysteresis_semitones = 0.35
    min_duration_sec = params["min_note_frames"] * frame_duration

    # Spectral flux for onset/silence gating
    stft = np.abs(librosa.stft(base_y, n_fft=1024, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.square(np.diff(stft, axis=1).clip(min=0)), axis=0))
    flux_threshold = params["onset_threshold_factor"] * np.median(np.abs(flux)) if flux.size else 0.0

    hmm_states = _decode_hmm_states(timeline, frame_duration, params["min_note_frames"])
    tuning_offset_cents = _estimate_tuning_offset(timeline)

    notes: List[NoteEvent] = []
    current: dict | None = None
    silence_run = 0
    filtered_pitch = None

    for idx, (frame, state) in enumerate(zip(timeline, hmm_states)):
        pitch = frame.pitch_hz if frame.pitch_hz > 0 else 0.0
        midi_val = float(librosa.hz_to_midi(pitch)) if pitch > 0 else None
        conf = float(frame.confidence)
        frame_flux = flux[idx] if idx < len(flux) else 0.0
        is_silent_frame = state == "silence" or frame_flux < flux_threshold or conf < params["conf_min"]

        if is_silent_frame:
            silence_run += 1
            filtered_pitch = None
            if current and silence_run >= silence_frames_required:
                duration_sec = current["end_sec"] - current["start_sec"] + frame_duration
                if duration_sec < min_duration_sec and notes:
                    # Merge into previous note to avoid staccato fragments
                    prev = notes[-1]
                    current_pitch_avg = current["pitch_sum"] / max(current["conf_sum"], 1e-9)
                    prev_dur = prev.end_sec - prev.start_sec
                    total_dur = prev_dur + duration_sec
                    weighted_pitch = (
                        prev.pitch_hz * prev_dur + current_pitch_avg * duration_sec
                    ) / max(total_dur, 1e-9)
                    prev.end_sec = current["end_sec"] + frame_duration
                    prev.pitch_hz = float(weighted_pitch)
                    prev.confidence = max(prev.confidence, current["max_conf"])
                    prev.velocity = max(prev.velocity, current["max_velocity"])
                else:
                    midi_avg = current["midi_sum"] / max(current["duration_frames"], 1)
                    midi_note = int(round(midi_avg - tuning_offset_cents / 100.0))
                    pitch_hz_avg = current["pitch_sum"] / max(current["conf_sum"], 1e-9)
                    notes.append(
                        NoteEvent(
                            start_sec=current["start_sec"],
                            end_sec=current["end_sec"] + frame_duration,
                            midi_note=midi_note,
                            pitch_hz=float(pitch_hz_avg),
                            confidence=float(current["max_conf"]),
                            velocity=float(current["max_velocity"]),
                            amplitude=float(current["amplitude_sum"] / max(current["duration_frames"], 1)),
                        )
                    )
                current = None
            continue

        silence_run = 0

        if midi_val is None:
            continue

        if filtered_pitch is None:
            filtered_pitch = midi_val
        else:
            delta = midi_val - filtered_pitch
            if abs(delta) > hysteresis_semitones:
                filtered_pitch += delta * 0.5
            filtered_pitch = 0.85 * filtered_pitch + 0.15 * midi_val

        stable_midi = filtered_pitch if state == "stable" else midi_val
        velocity = _map_velocity(rms[idx], rms_floor, rms_ceiling)

        if current is None:
            current = {
                "start_sec": frame.time,
                "end_sec": frame.time,
                "pitch_sum": pitch * conf,
                "conf_sum": conf,
                "max_conf": conf,
                "midi_sum": stable_midi,
                "duration_frames": 1,
                "max_velocity": velocity,
                "amplitude_sum": float(rms[idx]),
            }
            continue

        midi_jump = abs(stable_midi - current["midi_sum"] / max(current["duration_frames"], 1))
        if midi_jump > 1.5:
            # End current note and start a new one
            midi_avg = current["midi_sum"] / max(current["duration_frames"], 1)
            midi_note = int(round(midi_avg - tuning_offset_cents / 100.0))
            pitch_hz_avg = current["pitch_sum"] / max(current["conf_sum"], 1e-9)
            notes.append(
                NoteEvent(
                    start_sec=current["start_sec"],
                    end_sec=current["end_sec"] + frame_duration,
                    midi_note=midi_note,
                    pitch_hz=float(pitch_hz_avg),
                    confidence=float(current["max_conf"]),
                    velocity=float(current["max_velocity"]),
                    amplitude=float(current["amplitude_sum"] / max(current["duration_frames"], 1)),
                )
            )
            current = {
                "start_sec": frame.time,
                "end_sec": frame.time,
                "pitch_sum": pitch * conf,
                "conf_sum": conf,
                "max_conf": conf,
                "midi_sum": stable_midi,
                "duration_frames": 1,
                "max_velocity": velocity,
                "amplitude_sum": float(rms[idx]),
            }
            continue

        current["end_sec"] = frame.time
        current["pitch_sum"] += pitch * conf
        current["conf_sum"] += conf
        current["max_conf"] = max(current["max_conf"], conf)
        current["midi_sum"] += stable_midi
        current["duration_frames"] += 1
        current["max_velocity"] = max(current["max_velocity"], velocity)
        current["amplitude_sum"] += float(rms[idx])

    if current is not None:
        midi_avg = current["midi_sum"] / max(current["duration_frames"], 1)
        midi_note = int(round(midi_avg - tuning_offset_cents / 100.0))
        pitch_hz_avg = current["pitch_sum"] / max(current["conf_sum"], 1e-9)
        duration_sec = current["end_sec"] - current["start_sec"] + frame_duration
        if duration_sec >= min_duration_sec or not notes:
            notes.append(
                NoteEvent(
                    start_sec=current["start_sec"],
                    end_sec=current["end_sec"] + frame_duration,
                    midi_note=midi_note,
                    pitch_hz=float(pitch_hz_avg),
                    confidence=float(current["max_conf"]),
                    velocity=float(current["max_velocity"]),
                    amplitude=float(current["amplitude_sum"] / max(current["duration_frames"], 1)),
                )
            )
        elif notes:
            prev = notes[-1]
            current_pitch_avg = current["pitch_sum"] / max(current["conf_sum"], 1e-9)
            prev_dur = prev.end_sec - prev.start_sec
            total_dur = prev_dur + duration_sec
            weighted_pitch = (prev.pitch_hz * prev_dur + current_pitch_avg * duration_sec) / max(total_dur, 1e-9)
            prev.end_sec = current["end_sec"] + frame_duration
            prev.pitch_hz = float(weighted_pitch)
            prev.confidence = max(prev.confidence, current["max_conf"])
            prev.velocity = max(prev.velocity, current["max_velocity"])

    meta.pitch_tracker = "ensemble"
    chords = _detect_chords_from_chroma(base_y, base_sr, hop_length)

    return timeline, notes, chords
