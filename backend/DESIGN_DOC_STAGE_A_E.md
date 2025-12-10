# Stage A–E Transcription Pipeline and Evaluation

## Overview
The system processes audio through a five-stage pipeline (A→E) that extracts pitch, segments notes, applies theory, renders notation, and iteratively refines configuration based on benchmark metrics.

## Stage A – Preprocessing
- Load audio at the configured `sample_rate`, falling back to `fallback_sample_rate` when needed.
- Normalize loudness toward `target_lufs` while observing `silence_threshold_dbfs` to trim leading/trailing silence.
- Capture metadata such as detected tempo, beat overrides, and RMS/peak levels for downstream decisions.
- Optionally run HTDemucs source separation (GPU/MPS/CPU selectable) to produce `vocals`, `bass`, `drums`, and `other` stems;
  stems are loudness-normalized to the Stage A target and persisted into metadata for downstream routing.

## Stage B – Feature Extraction and Pitch Tracking
- Compute spectral features with `frame_length` / `hop_length` and median smoothing (`median_window`).
- Run multiple trackers (SwiftF0, YIN, CREPE/RMVPE/CQT when enabled) and blend them using `ensemble_weights` and `confidence_floor`.
- Derive frame-level MIDI estimates and confidences, respecting `fmin` / `fmax` and a `min_duration_sec` guard to suppress micro-events.

## Stage C – Segmentation and Grid Alignment
- Convert frame tracks into `NoteEvent` objects with `start_sec` / `end_sec`, `midi_note`, `pitch_hz`, and `confidence`.
- Enforce minimum lengths (`min_note_frames`, `min_note_ticks`) and merge notes separated by gaps shorter than `merge_gap_beats`.
- Clamp pitches into the readable range (24–108) while honoring `min_staccato_ticks` and quantization grid spacing (`ppq`, `bar_ticks`, `min_grid_ms`).

## Stage D – Theory, Quantization, and Rendering
- Apply harmonic context (key detection, chords) and optional beat/tempo overrides to quantize events to the musical grid.
- Render quantized notes to MusicXML and MIDI while preserving dynamics, articulations, and velocities.
- Exported assets accompany benchmark runs for inspection.

## Stage E – Evaluation and Refinement
- Benchmarks compute the full metric suite from `backend.metrics.compute_metrics`:
  - Harmonic Mean (HM) between Raw Pitch Accuracy (RPA) and OnsetF
  - RPA, Voicing Precision/Recall, Cents Accuracy (CA), Octave Accuracy (OA), Gross Error Accuracy (GEA)
  - Onset/Offset precision, recall, F1, plus combined OnsetOffsetF
- Benchmark outputs live under `benchmarks_results/<benchmark_name>/<audio>.json` with per-suite means in `suite_summary.json` and dataset-wide rollups in `global_summary.json`.
- Noise robustness (Level 05) evaluates SNR20/10/0 variants to track degradation slopes.
- `backend/refinement.RefinementLoop` reads metric JSON and, when HM or onset quality falls below targets, increases SwiftF0 weight, median smoothing, `confidence_floor`, `min_note_frames`, and onset gating. CA/OA drops trigger wider note filtering and frequency-band shifts (raise `fmin`, lower `fmax`). Noise-related HM gaps tighten smoothing and raise Swift weighting further.
- Every refinement call appends to `benchmarks_results/refinement_history.jsonl`, capturing timestamped configs, metric snapshots, and applied changes.

## Detector Behaviors and Overrides
- Trackers prioritize SwiftF0 for robustness; weights are shifted toward Swift when HM lags or noise robustness declines while reducing YIN emphasis.
- Beat/tempo overrides (`tempo_override`, `beat_times_override`) propagate into quantization to stabilize timing in challenging material.
- Onset thresholds and `confidence_floor` rise when onset F1 or noise metrics underperform.

## Ensemble Logic
- Weighted sum of tracker confidences forms the fused pitch contour; `confidence_floor` zeroes out low-confidence frames.
- Median filtering (`median_window`) and `merge_gap_beats` remove jitter and stitch near-adjacent notes.
- `min_note_frames` prevents spurious short notes, with adaptive increments driven by CA/OA or noise degradation.

## Segmentation and Quantization Rules
- Notes shorter than `min_duration_sec` or `min_note_frames` are suppressed.
- Quantization respects `ppq` and `bar_ticks`; durations shorter than `min_note_ticks` are lengthened to readable values.
- Grid spacing never drops below `min_grid_ms` to avoid excessively dense notation.

## Metrics and Tolerances
- Tolerances come from `config.json`: `onset_tolerance_sec`, `offset_tolerance_sec`, `pitch_tolerance_cents`, and `gross_error_cents`.
- Metric fields are always populated; suite summaries average each metric across files and include HM, Onset/Offset F1, Voicing, CA, OA, and GEA.

## Benchmark Levels and Noise Suites
- Levels 01–04 mirror the clean datasets (scales, simple melodies, melody+chords, pop loops).
- Level 05 auto-generates white-noise mixes at 20/10/0 dB SNR for all Level 01–03 clips via `utils.audio_noise.add_noise_snr` and `ensure_noisy_audio`.
- Results reside in `benchmarks_results/noise_robustness/SNR{level}/suite_summary.json` and fold into `global_summary.json` under `datasets.05_noise_robustness`.

## Refinement Loop Flow
1. Load metrics JSON (suite or global summary) and flatten dataset metrics when present.
2. Compare HM/OnsetF/CA/OA against targets; compute noise gaps relative to SNR20.
3. Propose updates: boost Swift weight, reduce YIN, raise `confidence_floor`, expand `median_window`, tighten onset thresholds, and shift `fmin`/`fmax` as needed.
4. Persist updates to `config.json` (unless `--dry-run`) and append a JSONL record to `refinement_history.jsonl` with metrics_before/after and applied changes.

## Sample Config Snippet
```json
{
  "median_window": 11,
  "confidence_floor": 0.1,
  "ensemble_weights": {"swift": 0.45, "yin": 0.15, "crepe": 0.25},
  "onset_tolerance_sec": 0.05,
  "offset_tolerance_sec": 0.08,
  "pitch_tolerance_cents": 50.0,
  "gross_error_cents": 100.0,
  "target_hm": 0.9,
  "target_onset_f": 0.9
}
```
