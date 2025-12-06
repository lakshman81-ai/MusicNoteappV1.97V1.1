# MP3-to-Music-Note Pipeline Review

This document summarizes the existing conversion pipeline and proposes targeted improvements.

## Current Flow (Stages A→D)
1. **Stage A – Load & Preprocess (`pipeline/stage_a.py`)**
   - Reads audio via `soundfile` with a `librosa` fallback, mixes to mono, trims silence, resamples to 22.05 kHz, and normalizes loudness toward −20 LUFS. Clips shorter than 0.5 s are rejected.
2. **Stage B – Feature Extraction (`pipeline/stage_b.py`)**
   - Estimates pitch using pyin/yin by default with optional CREPE, then refines F0 via harmonic summation, converts frames to note events, estimates chords, and annotates alternative tunings.
3. **Stage C – Theory Application (`pipeline/stage_c.py`)**
   - Injects music-theory labels (key, scale degrees, chord quality) and aligns note/chord metadata with the detected timeline.
4. **Stage D – Quantize & Render (`pipeline/stage_d.py`)**
   - Quantizes events, infers tempo/beat grid, renders MusicXML, and returns MIDI bytes via `_musicxml_to_midi_bytes` in `transcription.py`.
5. **Client-side Fallback (`services/transcriptionService.ts`)**
   - When running in-browser, audio is preprocessed (trim, resample, normalize) and notes/chords are converted to MusicXML if the backend pipeline is unavailable.

## Observed Strengths
- Silence trimming, minimum-duration guardrails, and LUFS normalization reduce empty/quiet inputs before pitch tracking.
- Harmonic-summation refinement and alternative-tuning annotations improve F0 robustness beyond raw pyin output.
- MusicXML rendering is available both server-side and client-side, keeping the UI responsive even without the backend.

## Suggested Improvements
1. **Stricter validation of silent or clipped inputs**
   - After `_normalize_loudness`, explicitly reject audio whose peak remains near zero or whose RMS falls below a floor (e.g., −50 dBFS) to avoid propagating near-silence into Stage B.

2. **Adaptive pitch-tracker selection**
   - Auto-select CREPE for polyphonic or low-SNR audio when the dependency is present, while keeping pyin/yin for monophonic cases to reduce compute.
   - Surface the chosen tracker in metadata so downstream consumers (UI/tests) know which algorithm produced the notes.

3. **Tempo/beat estimation resilience**
   - In Stage D, fall back to a stable default (e.g., 120 BPM with measure-level beat times) when beat tracking fails instead of continuing with `None`, reducing quantization artifacts.

4. **Confidence-aware filtering and labeling**
   - In Stage B, discard or down-weight note candidates whose voiced probability falls below a threshold; plumb these confidences into the MusicXML as `<note>` annotations to aid UI heatmaps.

5. **Consistent MIDI rendering**
   - Replace the temporary-file hop in `_musicxml_to_midi_bytes` with an in-memory buffer (e.g., `io.BytesIO`) to avoid filesystem churn and make the function safe for concurrent calls.

6. **UI feedback for preprocessing errors**
   - In `services/transcriptionService.ts`, map preprocessing errors (e.g., "Audio is silent" or "too short") to user-friendly toasts so users know why a file failed before backend transcription.

7. **Benchmark hooks**
   - Extend `benchmarks/benchmark_local_file.py` to log per-stage timings and tracker selection, enabling regression detection when adjusting preprocessing or pitch tracking.

## Quick Wins to Prioritize
- Add RMS/peak validation post-normalization (Improvement 1).
- Emit tracker choice and confidence-based filtering (Improvements 2 & 4) to better explain UI results without large architectural changes.
- Convert MIDI rendering to use `io.BytesIO` (Improvement 5) to simplify deployment.
