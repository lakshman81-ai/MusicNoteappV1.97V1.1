from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

import librosa
import numpy as np
from scipy import linalg, signal

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
    compressed_full = np.interp(np.arange(len(acf)), compressed_time, compressed, left=compressed[0], right=0.0)
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
        high_band = signal.lfilter(high_b, high_a, whitened)
        high_band = np.maximum(high_band, 0.0)

        ac_low = librosa.autocorrelate(low_band)
        ac_high = librosa.autocorrelate(high_band)
        ac_sum = ac_low + ac_high

        if not np.any(np.isfinite(ac_sum)):
            continue

        ac_sum = ac_sum / (np.max(np.abs(ac_sum)) + 1e-9)
        ac_sum = _apply_octave_pruning(ac_sum, octave_compression, octave_suppression)

        ac_sum[:min_lag] = 0.0
        ac_sum[max_lag:] = 0.0

        peak_idx = int(np.argmax(ac_sum))
        peak_val = float(ac_sum[peak_idx]) if peak_idx > 0 else 0.0
        salience[i] = peak_val

        if peak_idx <= 0 or peak_val < peak_threshold:
            continue

        pitches[i] = sr / float(peak_idx)
        conf[i] = float(np.clip(peak_val, 0.0, 1.0))

    return pitches, conf, salience


def _detector_swift(y: np.ndarray, sr: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray] | None:
    if importlib.util.find_spec("swift" ) is None and importlib.util.find_spec("swiftf0") is None:
        return None
    try:
        swift = importlib.import_module("swift").SwiftF0
    except Exception:
        try:
            swift = importlib.import_module("swiftf0").SwiftF0
        except Exception:
            return None

    cfg = get_config()
    target_sr = 16000
    pipeline_sr = int(cfg.get("sample_rate", sr))
    already_at_target_sr = sr == target_sr or (pipeline_sr == target_sr and sr == pipeline_sr)

    if already_at_target_sr:
        y_swift = y
        swift_sr = sr
        swift_hop = hop_length
    else:
        y_swift = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        swift_sr = target_sr
        swift_hop = int(round(hop_length * target_sr / sr))

    model = swift(sample_rate=swift_sr)
    result = model.infer(y_swift, hop_size=swift_hop)
    pitches = np.asarray(result["f0_hz"], dtype=float)
    conf = np.asarray(result.get("confidence", np.ones_like(pitches)), dtype=float)

    swift_times = np.arange(len(pitches)) * (swift_hop / float(swift_sr))
    target_frames = int(np.ceil(len(y) / hop_length))
    target_times = np.arange(target_frames) * (hop_length / float(sr))

    aligned_pitches = np.interp(target_times, swift_times, pitches, left=np.nan, right=np.nan)
    aligned_conf = np.interp(target_times, swift_times, conf, left=0.0, right=0.0)

    return aligned_pitches, aligned_conf


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
    """Stage B: ensemble pitch detection with deterministic rules."""
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

    swift_output = _detector_swift(swift_y, swift_sr, hop_length)
    if swift_output is not None:
        detector_outputs["swift"] = swift_output

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
        salience = float(autocorr_salience[idx]) if idx < len(autocorr_salience) else 0.0
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

    # Stage C rules: deterministic segmentation
    frame_duration = float(hop_length / sr)
    silence_frames_required = int(np.ceil(0.12 / frame_duration))

    # Spectral flux for onset/silence gating
    stft = np.abs(librosa.stft(base_y, n_fft=1024, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.square(np.diff(stft, axis=1).clip(min=0)), axis=0))
    flux_threshold = params["onset_threshold_factor"] * np.median(np.abs(flux)) if flux.size else 0.0

    notes: List[NoteEvent] = []
    current_note: NoteEvent | None = None
    stable_run = 0
    low_conf_run = 0
    silence_run = 0

    def cents_diff(a: float, b: float) -> float:
        return abs(1200.0 * np.log2(a / b)) if a > 0 and b > 0 else 0.0

    for idx, frame in enumerate(timeline):
        pitch = frame.pitch_hz
        confidence = frame.confidence
        is_unstable = unstable[idx]
        frame_flux = flux[idx] if idx < len(flux) else 0.0
        is_silent_frame = frame_flux < flux_threshold

        if np.isfinite(pitch) and pitch > 0 and not is_unstable:
            if current_note and cents_diff(pitch, current_note.pitch_hz) <= 30.0:
                stable_run += 1
            elif current_note is None:
                stable_run = 1
            else:
                stable_run = 0
        else:
            stable_run = 0

        if confidence < params["conf_min"]:
            low_conf_run += 1
        else:
            low_conf_run = 0

        silence_run = silence_run + 1 if is_silent_frame else 0

        if current_note is None:
            if stable_run >= params["min_note_frames"] and confidence >= params["conf_min"]:
                velocity = _map_velocity(rms[idx], rms_floor, rms_ceiling)
                current_note = NoteEvent(
                    start_sec=frame.time,
                    end_sec=frame.time,
                    midi_note=frame.midi or int(round(librosa.hz_to_midi(pitch))) if pitch > 0 else 0,
                    pitch_hz=pitch,
                    confidence=float(confidence),
                    velocity=velocity,
                    amplitude=float(rms[idx]),
                )
                stable_run = 0
            continue

        # termination conditions
        pitch_jump = cents_diff(pitch, current_note.pitch_hz) if pitch > 0 else 0.0
        end_due_to_jump = pitch > 0 and pitch_jump > 120.0
        end_due_to_conf = low_conf_run >= params["min_note_frames"]
        end_due_to_silence = silence_run >= silence_frames_required

        if end_due_to_jump or end_due_to_conf or end_due_to_silence:
            current_note.end_sec = frame.time
            notes.append(current_note)
            current_note = None
            low_conf_run = 0
            silence_run = 0
            stable_run = 0
            continue

        current_note.end_sec = frame.time
        current_note.confidence = float(max(current_note.confidence, confidence))
        current_note.pitch_hz = float(pitch if pitch > 0 else current_note.pitch_hz)
        current_note.midi_note = current_note.midi_note if current_note.midi_note else (frame.midi or current_note.midi_note)
        current_note.velocity = _map_velocity(rms[idx], rms_floor, rms_ceiling)
        current_note.amplitude = float(rms[idx])

    if current_note is not None:
        current_note.end_sec = float(times[-1]) if len(times) else current_note.end_sec
        notes.append(current_note)

    meta.pitch_tracker = "ensemble"
    chords = _detect_chords_from_chroma(base_y, base_sr, hop_length)

    return timeline, notes, chords
