from __future__ import annotations

import importlib
from typing import List, Tuple

import librosa
import numpy as np
from sklearn.cluster import KMeans
from music21 import note as m21note, stream

from .models import (
    MetaData,
    FramePitch,
    NoteEvent,
    ChordEvent,
    AlternativePitch,
)


def _crepe_available() -> bool:
    return importlib.util.find_spec("crepe") is not None


def _pitch_with_pyin(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    min_voiced_confidence: float = 0.1,
    unvoiced_confidence: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    voiced_probs = np.nan_to_num(voiced_probs, nan=0.0)
    min_voiced_confidence = max(0.0, min_voiced_confidence)
    voiced_probs = np.clip(voiced_probs, 0.0, 1.0)
    voiced_probs = np.where(
        voiced_flag, np.maximum(voiced_probs, min_voiced_confidence), unvoiced_confidence
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0, voiced_flag, voiced_probs


def _pitch_with_crepe(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    voiced_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crepe = importlib.import_module("crepe")
    step_size_ms = hop_length * 1000.0 / sr
    if sr != 16000:
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr_used = 16000
    else:
        y_resampled = y
        sr_used = sr

    time, frequency, confidence, _ = crepe.predict(
        y_resampled, sr_used, step_size=step_size_ms, viterbi=True
    )
    voiced_threshold = max(0.0, voiced_threshold)
    voiced_flag = confidence >= voiced_threshold
    return time.astype(float), frequency.astype(float), voiced_flag, confidence.astype(float)


def _harmonic_summation_refine(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    times: np.ndarray,
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    n_fft: int = 2048,
    harmonics: int = 4,
) -> np.ndarray:
    """Refine F0 using harmonic summation around the detected pitch."""

    if not np.any(voiced_mask):
        return f0

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    refined = np.copy(f0)
    for idx, (time, base_f0, is_voiced) in enumerate(zip(times, f0, voiced_mask)):
        if not is_voiced or base_f0 is None or np.isnan(base_f0) or base_f0 <= 0:
            continue

        frame_idx = int(
            np.clip(
                np.round(librosa.time_to_frames(time, sr=sr, hop_length=hop_length)),
                0,
                S.shape[1] - 1,
            )
        )

        candidate_freqs = base_f0 * (2 ** np.linspace(-0.25, 0.25, 5))
        best_freq = base_f0
        best_score = -np.inf

        for candidate in candidate_freqs:
            if candidate <= 0:
                continue

            score = 0.0
            for h in range(1, harmonics + 1):
                freq = candidate * h
                if freq > freqs[-1]:
                    break
                mag = float(np.interp(freq, freqs, S[:, frame_idx]))
                score += mag / h

            if score > best_score:
                best_score = score
                best_freq = candidate

        refined[idx] = best_freq

    return refined


def _apply_voicing_hysteresis(
    voiced_probs: np.ndarray, on_threshold: float = 0.55, off_threshold: float = 0.35
) -> np.ndarray:
    """Convert probabilities into a stable voiced mask using hysteresis."""

    mask = np.zeros_like(voiced_probs, dtype=bool)
    active = False
    for i, prob in enumerate(voiced_probs):
        if active:
            if prob <= off_threshold:
                active = False
        else:
            if prob >= on_threshold:
                active = True
        mask[i] = active
    return mask


def _smooth_midi_with_voicing(
    f0: np.ndarray,
    voiced_probs: np.ndarray,
    prob_threshold: float = 0.45,
    smoothing_window: int = 13,
) -> np.ndarray:
    if smoothing_window < 11 or smoothing_window > 17 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be an odd integer between 11 and 17")

    midi = librosa.hz_to_midi(f0)
    midi = np.where(np.isfinite(midi), midi, np.nan)
    smoothed = midi.copy()

    voiced_mask = (voiced_probs >= prob_threshold) & np.isfinite(midi)

    half = smoothing_window // 2
    for i in range(len(midi)):
        if not voiced_mask[i]:
            continue
        start = max(0, i - half)
        end = min(len(midi), i + half + 1)

        window_mask = voiced_mask[start:end]
        if not np.any(window_mask):
            continue

        smoothed[i] = float(np.median(midi[start:end][window_mask]))

    return smoothed


def _run_basic_pitch(y: np.ndarray, sr: int) -> Tuple[List[FramePitch], List[NoteEvent]]:
    """Run Basic-Pitch if installed; raises on failure so the caller can fallback."""

    if not _basic_pitch_available():
        raise RuntimeError("basic_pitch not available")

    basic_pitch = importlib.import_module("basic_pitch.inference")
    icassp_path = importlib.import_module("basic_pitch").ICASSP2022_MODEL_PATH

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        tmp_path = tmp.name

    try:
        outputs, _, _ = basic_pitch.predict([tmp_path], model_path=icassp_path, output_prediction=False)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    note_events = outputs[0]["note_events"] if outputs else []
    timeline: List[FramePitch] = []
    notes: List[NoteEvent] = []
    for start, end, midi_note, amplitude in note_events:
        pitch_hz = float(librosa.midi_to_hz(midi_note))
        notes.append(
            NoteEvent(
                start_sec=float(start),
                end_sec=float(end),
                midi_note=int(midi_note),
                pitch_hz=pitch_hz,
                confidence=float(amplitude),
                velocity=float(np.clip(amplitude, 0.0, 1.0) * 105.0),
                amplitude=float(np.clip(amplitude, 0.0, 1.0)),
            )
        )
        timeline.append(
            FramePitch(
                time=float(start),
                pitch_hz=pitch_hz,
                midi=int(midi_note),
                confidence=float(np.clip(amplitude, 0.0, 1.0)),
            )
        )
    return timeline, notes


def _estimate_polyphonic_peaks(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    target_frames: int,
    bins_per_octave: int = 24,
    peak_spread_db: float = 18.0,
    top_k: int = 3,
) -> List[List[int]]:
    """Estimate secondary pitch peaks per frame using a CQT front-end.

    This lightweight pass is intended to surface additional stable pitch
    candidates that often occur in polyphonic passages but are suppressed by
    monophonic trackers (e.g., pYIN/CREPE). We only keep peaks within ``peak_spread_db``
    of the frame maximum to avoid noise and return up to ``top_k`` MIDI candidates
    per frame.
    """

    n_bins = int(np.ceil(12 * np.log2(fmax / fmin))) + 1
    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        pad_mode="reflect",
    )

    mags_db = librosa.amplitude_to_db(np.abs(cqt) + 1e-6, ref=np.max)
    fmin_midi = float(librosa.hz_to_midi(fmin))
    step = 12.0 / bins_per_octave

    frame_peaks: List[List[int]] = []
    for frame_idx in range(mags_db.shape[1]):
        frame = mags_db[:, frame_idx]
        if not np.any(np.isfinite(frame)):
            frame_peaks.append([])
            continue

        max_db = float(np.max(frame))
        threshold = max_db - peak_spread_db
        candidate_idx = np.where(frame >= threshold)[0]
        if candidate_idx.size == 0:
            frame_peaks.append([])
            continue

        sorted_idx = candidate_idx[np.argsort(frame[candidate_idx])[::-1]]
        unique_midis: List[int] = []
        for idx in sorted_idx[:top_k * 2]:
            midi_val = int(round(fmin_midi + idx * step))
            if midi_val in unique_midis:
                continue
            unique_midis.append(midi_val)
            if len(unique_midis) >= top_k:
                break
        frame_peaks.append(unique_midis)

    if len(frame_peaks) < target_frames:
        frame_peaks.extend([[] for _ in range(target_frames - len(frame_peaks))])
    return frame_peaks[:target_frames]


def _assign_voices(notes: List[NoteEvent]) -> List[NoteEvent]:
    if len(notes) < 2:
        for note in notes:
            note.voice = "voice1"
        return notes

    pitches = np.array([n.midi_note for n in notes], dtype=float).reshape(-1, 1)
    try:
        clusters = KMeans(n_clusters=2, n_init="auto", random_state=0).fit_predict(pitches)
        centers = np.array([pitches[clusters == i].mean() for i in range(2)])
    except Exception:
        for note in notes:
            note.voice = "voice1"
        return notes

    ordering = np.argsort(centers)
    label_to_voice = {ordering[1]: "voice1", ordering[0]: "voice2"}
    for label, note in zip(clusters, notes):
        note.voice = label_to_voice.get(label, "voice1")
    return notes


def _detect_chords_from_chroma(
    y: np.ndarray, sr: int, hop_length: int, tempo_bpm: float | None = None
) -> List[ChordEvent]:
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
        beat_val = float(time_sec * (tempo_bpm or 120.0) / 60.0)
        symbol = f"{best_root}{'' if best_quality == 'maj' else 'm'}"
        chords.append(ChordEvent(time=time_sec, beat=beat_val, symbol=symbol, root=best_root, quality=best_quality))
    return chords


def _build_timeline(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
    midi_values: np.ndarray | None,
    rms: np.ndarray | None,
    min_confidence: float,
    rms_floor: float | None,
) -> List[FramePitch]:
    timeline: List[FramePitch] = []
    for idx, (t, hz, is_voiced, conf) in enumerate(
        zip(times, f0, voiced_flag, voiced_probs)
    ):
        if not is_voiced or hz is None or np.isnan(hz):
            timeline.append(FramePitch(time=float(t), pitch_hz=0.0, midi=None, confidence=float(conf)))
            continue
        midi_value = midi_values[idx] if midi_values is not None else librosa.hz_to_midi(float(hz))
        if np.isnan(midi_value):
            timeline.append(FramePitch(time=float(t), pitch_hz=0.0, midi=None, confidence=float(conf)))
            continue
        midi = int(round(float(midi_value)))
        timeline.append(
            FramePitch(
                time=float(t),
                pitch_hz=float(hz),
                midi=midi,
                confidence=float(conf),
            )
        )
    return timeline


def _segment_notes_from_timeline(
    timeline: List[FramePitch],
    frame_duration: float,
    min_duration: float = 0.06,
    max_duration: float | None = 8.0,
    rest_threshold: float = 0.08,
    median_window: int = 5,
    alt_pitch_frames: List[List[int]] | None = None,
    rms: np.ndarray | None = None,
    rms_floor: float | None = None,
    velocity_range: tuple[int, int] = (20, 105),
    velocity_humanization: float | None = None,
    velocity_seed: int | None = None,
    min_confidence: float = 0.12,
) -> List[NoteEvent]:
    """Segment notes using a voiced/unvoiced HMM with Viterbi decoding.

    The state space consists of an explicit unvoiced/rest state and a set of
    MIDI bins derived from the observed timeline. Emission probabilities use
    smoothed MIDI/confidence estimates, while transition costs adapt to local
    variability (MAD) to discourage implausible jumps. The decoded state path is
    post-processed with duration constraints and gap-aware merging.
    """

    if not timeline:
        return []

    times = np.array([frame.time for frame in timeline], dtype=float)
    midi_obs = np.array([frame.midi if frame.midi is not None else np.nan for frame in timeline], dtype=float)
    conf_obs = np.array([frame.confidence for frame in timeline], dtype=float)

    if median_window % 2 == 0:
        median_window += 1
    half = median_window // 2

    # Smooth MIDI observations with a median filter over voiced frames.
    smoothed_midi = midi_obs.copy()
    finite_mask = np.isfinite(midi_obs)
    global_median = float(np.nanmedian(midi_obs)) if np.any(finite_mask) else np.nan
    for idx in range(len(midi_obs)):
        start = max(0, idx - half)
        end = min(len(midi_obs), idx + half + 1)
        window = midi_obs[start:end]
        valid = window[np.isfinite(window)]
        if valid.size:
            smoothed_midi[idx] = float(np.median(valid))
        elif np.isfinite(global_median):
            smoothed_midi[idx] = global_median

    # Smooth confidence with a small moving average.
    kernel = np.ones(median_window, dtype=float)
    smoothed_conf = np.convolve(conf_obs, kernel / kernel.sum(), mode="same")

    if np.all(~np.isfinite(smoothed_midi)):
        return []

    vel_min, vel_max = velocity_range
    humanize_amount = None
    rng = None
    if velocity_humanization is not None and velocity_humanization > 0:
        humanize_amount = float(min(velocity_humanization, 5.0))
        rng = np.random.default_rng(velocity_seed)

    observed_midis = smoothed_midi[np.isfinite(smoothed_midi)]
    midi_min = int(np.floor(np.min(observed_midis))) - 1
    midi_max = int(np.ceil(np.max(observed_midis))) + 1
    midi_states = list(range(midi_min, midi_max + 1))
    states: List[int | None] = [None] + midi_states

    # Transition penalty scale based on local variability.
    midi_diffs = np.diff(observed_midis) if observed_midis.size > 1 else np.array([0.0])
    mad = float(np.median(np.abs(midi_diffs - np.median(midi_diffs)))) if midi_diffs.size else 0.0
    jump_scale = max(mad, 0.5)

    sigma = 0.9
    tiny = 1e-6

    def emission_for_state(state: int | None, midi_value: float, conf_value: float) -> float:
        if state is None:
            return float(np.log(max(1.0 - conf_value, tiny)))
        if not np.isfinite(midi_value):
            distance = 3.0
        else:
            distance = abs(midi_value - state)
        likelihood = -0.5 * (distance / sigma) ** 2
        return likelihood + float(np.log(conf_value + tiny))

    # Build emission matrix (states x frames).
    emissions = np.zeros((len(states), len(timeline)), dtype=float)
    for t_idx, (midi_val, conf_val) in enumerate(zip(smoothed_midi, smoothed_conf)):
        for s_idx, state in enumerate(states):
            emissions[s_idx, t_idx] = emission_for_state(state, midi_val, conf_val)

    def transition_cost(prev: int | None, nxt: int | None) -> float:
        if prev is None and nxt is None:
            return 0.0
        if prev is None or nxt is None:
            return -0.2
        if prev == nxt:
            return 0.05
        distance = abs(prev - nxt)
        return -distance / (jump_scale + tiny)

    # Precompute transition matrix for efficiency.
    transition_matrix = np.zeros((len(states), len(states)), dtype=float)
    for i, prev_state in enumerate(states):
        for j, next_state in enumerate(states):
            transition_matrix[i, j] = transition_cost(prev_state, next_state)

    # Viterbi decoding.
    log_probs = np.full((len(states), len(timeline)), -np.inf, dtype=float)
    backptr = np.full((len(states), len(timeline)), -1, dtype=int)

    log_probs[:, 0] = emissions[:, 0]
    for t_idx in range(1, len(timeline)):
        for curr_idx in range(len(states)):
            transition_candidates = log_probs[:, t_idx - 1] + transition_matrix[:, curr_idx]
            best_prev = int(np.argmax(transition_candidates))
            log_probs[curr_idx, t_idx] = transition_candidates[best_prev] + emissions[curr_idx, t_idx]
            backptr[curr_idx, t_idx] = best_prev

    best_last_state = int(np.argmax(log_probs[:, -1]))
    best_path: List[int | None] = [None] * len(timeline)
    best_path[-1] = states[best_last_state]
    for t_idx in range(len(timeline) - 1, 0, -1):
        best_last_state = backptr[best_last_state, t_idx]
        best_path[t_idx - 1] = states[best_last_state]

    # Convert the best path into contiguous segments.
    segments: List[Tuple[int | None, int, int]] = []
    seg_start = 0
    for idx in range(1, len(best_path)):
        if best_path[idx] != best_path[seg_start]:
            segments.append((best_path[seg_start], seg_start, idx))
            seg_start = idx
    segments.append((best_path[seg_start], seg_start, len(best_path)))

    # Enforce minimum duration by merging with neighbors.
    def segment_duration(start: int, end: int) -> float:
        return (times[end - 1] - times[start]) + frame_duration

    i = 0
    while i < len(segments):
        state, start_idx, end_idx = segments[i]
        duration = segment_duration(start_idx, end_idx)
        if duration >= min_duration or len(segments) == 1:
            i += 1
            continue

        if i == 0:
            segments[i + 1] = (segments[i + 1][0], start_idx, segments[i + 1][2])
        elif i == len(segments) - 1:
            prev_state, prev_start, prev_end = segments[i - 1]
            segments[i - 1] = (prev_state, prev_start, end_idx)
        else:
            prev_state, prev_start, prev_end = segments[i - 1]
            next_state, next_start, next_end = segments[i + 1]
            if prev_state == state:
                segments[i - 1] = (prev_state, prev_start, end_idx)
            elif next_state == state:
                segments[i + 1] = (next_state, start_idx, next_end)
            else:
                # Merge with the closer pitch neighbor when possible.
                prev_dist = abs(prev_state - state) if prev_state is not None and state is not None else np.inf
                next_dist = abs(next_state - state) if next_state is not None and state is not None else np.inf
                if prev_dist <= next_dist:
                    segments[i - 1] = (prev_state, prev_start, end_idx)
                else:
                    segments[i + 1] = (next_state, start_idx, next_end)
        segments.pop(i)

    # Optionally enforce a maximum duration by splitting long notes.
    if max_duration is not None and max_duration > 0:
        split_segments: List[Tuple[int | None, int, int]] = []
        for state, start_idx, end_idx in segments:
            duration = segment_duration(start_idx, end_idx)
            if state is None or duration <= max_duration:
                split_segments.append((state, start_idx, end_idx))
                continue

            max_frames = int(np.ceil(max_duration / frame_duration))
            current_start = start_idx
            while current_start < end_idx:
                current_end = min(end_idx, current_start + max_frames)
                split_segments.append((state, current_start, current_end))
                current_start = current_end
        segments = split_segments

    # Convert segments into notes while handling gap-aware merging.
    candidate_notes: List[NoteEvent] = []
    rms_peak = float(np.max(rms)) if rms is not None and rms.size else None
    for state, start_idx, end_idx in segments:
        if state is None:
            continue
        start_time = times[start_idx]
        end_time = times[end_idx - 1] + frame_duration
        duration = end_time - start_time
        if duration < min_duration:
            continue
        midi_values = smoothed_midi[start_idx:end_idx]
        midi_value = int(round(float(np.nanmedian(midi_values))))
        pitch_hz = float(librosa.midi_to_hz(midi_value))
        confidence = float(np.mean(conf_obs[start_idx:end_idx]))
        if confidence < min_confidence:
            continue
        velocity = float((vel_min + vel_max) / 2.0)
        amplitude_norm = 0.5
        if rms is not None and rms_peak is not None and rms_peak > 0:
            note_rms = float(np.mean(rms[start_idx:end_idx]))
            floor = rms_floor if rms_floor is not None else 0.0
            norm = np.clip((note_rms - floor) / max(rms_peak - floor, 1e-6), 0.0, 1.0)
            velocity = float(vel_min + norm * (vel_max - vel_min))
            amplitude_norm = float(norm)
        if humanize_amount is not None and rng is not None:
            velocity += float(rng.uniform(-humanize_amount, humanize_amount))
        velocity = float(np.clip(velocity, vel_min, vel_max))
        alternatives: List[AlternativePitch] = []
        if alt_pitch_frames is not None and alt_pitch_frames:
            window_alts = [p for frame in alt_pitch_frames[start_idx:end_idx] for p in frame if p != midi_value]
            if window_alts:
                unique, counts = np.unique(window_alts, return_counts=True)
                coverage = counts / max(1, end_idx - start_idx)
                for alt_midi, cov in sorted(zip(unique, coverage), key=lambda x: x[1], reverse=True):
                    if cov < 0.35:
                        continue
                    alternatives.append(AlternativePitch(midi=int(alt_midi), confidence=float(min(1.0, cov))))

        candidate_notes.append(
            NoteEvent(
                start_sec=float(start_time),
                end_sec=float(end_time),
                midi_note=midi_value,
                pitch_hz=pitch_hz,
                confidence=confidence,
                velocity=velocity,
                amplitude=amplitude_norm,
                alternatives=alternatives,
            )
        )

    if not candidate_notes:
        return []

    candidate_notes.sort(key=lambda n: n.start_sec)
    merged_notes: List[NoteEvent] = []
    merged_notes.append(candidate_notes[0])
    for note in candidate_notes[1:]:
        last = merged_notes[-1]
        gap = note.start_sec - last.end_sec
        if gap < rest_threshold and note.midi_note == last.midi_note:
            merged_notes[-1] = NoteEvent(
                start_sec=last.start_sec,
                end_sec=note.end_sec,
                midi_note=last.midi_note,
                pitch_hz=float(librosa.midi_to_hz(last.midi_note)),
                confidence=float((last.confidence + note.confidence) / 2.0),
                velocity=float((last.velocity + note.velocity) / 2.0),
                amplitude=float((last.amplitude + note.amplitude) / 2.0),
            )
        else:
            merged_notes.append(note)

    return merged_notes


def _mark_attacks(notes: List[NoteEvent], y: np.ndarray, sr: int, hop_length: int) -> List[NoteEvent]:
    try:
        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    except Exception:
        flux = np.array([])
    if flux.size == 0:
        for note in notes:
            note.articulation = "legato"
        return notes

    times = librosa.times_like(flux, sr=sr, hop_length=hop_length)
    threshold = float(np.percentile(flux, 70)) if flux.size else 0.0
    for note in notes:
        start_frame = int(np.argmin(np.abs(times - note.start_sec))) if times.size else 0
        energy = float(flux[start_frame]) if start_frame < flux.size else 0.0
        note.articulation = "articulated" if energy >= threshold else "legato"
    return notes


def _detect_key_signature(notes: List[NoteEvent]) -> str | None:
    if not notes:
        return None
    s = stream.Stream()
    for n in notes:
        m_note = m21note.Note(n.midi_note)
        duration_qL = float(n.duration_beats or max((n.end_sec - n.start_sec) * 2.0, 0.25))
        m_note.quarterLength = max(duration_qL, 0.25)
        s.append(m_note)
    try:
        key_obj = s.analyze("key")
        return key_obj.tonic.name + ("m" if key_obj.mode == "minor" else "")
    except Exception:
        return None


def extract_features(
    y: np.ndarray,
    sr: int,
    meta: MetaData,
    use_crepe: bool = False,
    velocity_humanization: float | None = None,
    velocity_seed: int | None = None,
    crepe_voiced_threshold: float = 0.5,
    pyin_min_confidence: float = 0.1,
    voicing_on_threshold: float = 0.55,
    voicing_off_threshold: float = 0.35,
    smoothing_prob_threshold: float = 0.45,
    smoothing_window: int = 13,
) -> Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent]]:
    """
    Stage B: pitch tracking and simple note segmentation.
    """

    hop_length = meta.hop_length or 256
    fmin = librosa.note_to_hz("A1")
    fmax = librosa.note_to_hz("C7")

    tracker_used = "unknown"
    timeline: List[FramePitch] = []
    notes: List[NoteEvent] = []

    analysis_success = False
    times = f0 = voiced_flag = voiced_probs = None

    prefer_crepe = use_crepe or _crepe_available()
    if prefer_crepe:
        try:
            times, f0, voiced_flag, voiced_probs = _pitch_with_crepe(
                y, sr, hop_length, fmin=fmin, fmax=fmax, voiced_threshold=crepe_voiced_threshold
            )
            tracker_used = "crepe"
            analysis_success = True

            try:
                y_harm, _ = librosa.effects.hpss(y)
                py_times, py_f0, _, py_probs = _pitch_with_pyin(
                    y_harm,
                    sr,
                    hop_length,
                    fmin=fmin,
                    fmax=fmax,
                    min_voiced_confidence=max(pyin_min_confidence, 0.05),
                )
                py_probs = np.nan_to_num(py_probs, nan=0.0)
                py_f0_interp = np.interp(
                    times, py_times, np.nan_to_num(py_f0, nan=0.0), left=np.nan, right=np.nan
                )
                py_prob_interp = np.interp(times, py_times, py_probs, left=0.0, right=0.0)
                low_conf_mask = voiced_probs < crepe_voiced_threshold
                f0 = np.where(low_conf_mask, py_f0_interp, f0)
                voiced_flag = (voiced_probs >= crepe_voiced_threshold) | (
                    (py_prob_interp >= pyin_min_confidence) & np.isfinite(py_f0_interp)
                )
                voiced_probs = np.where(low_conf_mask, py_prob_interp, voiced_probs)
            except Exception:
                pass
        except Exception:
            analysis_success = False

    if not analysis_success:
        try:
            y_harm, _ = librosa.effects.hpss(y)
            times, f0, voiced_flag, voiced_probs = _pitch_with_pyin(
                y_harm,
                sr,
                hop_length,
                fmin=fmin,
                fmax=fmax,
                min_voiced_confidence=max(pyin_min_confidence, 0.05),
            )
            tracker_used = "hpss-pyin"
            analysis_success = True
        except Exception:
            pass

    if not analysis_success:
        times, f0, voiced_flag, voiced_probs = _pitch_with_pyin(
            y,
            sr,
            hop_length,
            fmin=fmin,
            fmax=fmax,
            min_voiced_confidence=pyin_min_confidence,
        )
        tracker_used = "pyin"
        analysis_success = True

    voiced_mask = _apply_voicing_hysteresis(
        voiced_probs, on_threshold=voicing_on_threshold, off_threshold=voicing_off_threshold
    ) & voiced_flag
    refined_f0 = _harmonic_summation_refine(
        y, sr, hop_length, times, f0, voiced_mask, n_fft=2048, harmonics=4
    )
    midi_smoothed = _smooth_midi_with_voicing(
        refined_f0,
        voiced_probs,
        prob_threshold=smoothing_prob_threshold,
        smoothing_window=smoothing_window,
    )

    rms = librosa.feature.rms(y=y, frame_length=meta.window_size or 2048, hop_length=hop_length)[0]
    alt_pitch_frames = _estimate_polyphonic_peaks(
        y,
        sr,
        hop_length,
        fmin,
        fmax,
        target_frames=len(times),
    )

    if rms.size < len(times):
        pad_width = len(times) - rms.size
        rms = np.pad(rms, (0, pad_width), mode="edge")
    else:
        rms = rms[: len(times)]
    rms_floor = float(np.percentile(rms, 10)) if rms.size else None

    timeline = _build_timeline(
        times,
        refined_f0,
        voiced_mask,
        voiced_probs,
        midi_smoothed,
        rms=rms,
        min_confidence=0.0,
        rms_floor=rms_floor,
    )

    if len(timeline) >= 2:
        frame_duration = float(timeline[1].time - timeline[0].time)
    else:
        frame_duration = float(hop_length / sr)

    notes = _segment_notes_from_timeline(
        timeline,
        frame_duration=frame_duration,
        min_duration=0.06,
        max_duration=8.0,
        rest_threshold=0.08,
        median_window=5,
        alt_pitch_frames=alt_pitch_frames,
        rms=rms,
        rms_floor=rms_floor,
        velocity_humanization=velocity_humanization,
        velocity_seed=velocity_seed,
        min_confidence=max(0.1, pyin_min_confidence),
    )

    meta.pitch_tracker = tracker_used

    notes = _assign_voices(notes)
    chords = _detect_chords_from_chroma(y, sr, hop_length, tempo_bpm=meta.tempo_bpm)
    notes = _mark_attacks(notes, y, sr, hop_length)
    meta.detected_key = meta.detected_key or _detect_key_signature(notes)

    return timeline, notes, chords
