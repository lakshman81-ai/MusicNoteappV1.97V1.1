# Benchmark plan and refinement guide

This directory holds fixtures and scripts for evaluating the transcription pipeline end-to-end. Use the steps below to run the suite, interpret the metrics, and tune the quantization/tempo logic when accuracy drifts.

## Directory layout
```
benchmarks/
  01_scales/
    README.md          # scenario-specific notes and fixture checklist
    results.md         # recorded runs using the template below
    audio/             # raw audio fixtures (wav, mp3, etc.)
    references/        # MIDI or MusicXML ground truth when available
  02_simple_melodies/
  03_melody_plus_chords/
  04_pop_loops/
```
Add new scenarios by creating additional numbered folders that follow the same pattern (README + results.md + fixture subfolders). Keep corresponding reference files aligned by basename (e.g., `folk_tune_in_g.mid` and `folk_tune_in_g_reference.musicxml`).

## Prerequisites
- Python 3.10+ with `pip install -r backend/requirements.txt`.
- `ffmpeg` installed if you plan to benchmark MP3 fixtures.
- Audio/reference pairs placed under `benchmarks/audio/` and `benchmarks/reference/` (see defaults in `get_default_suite()` inside `benchmark_local_file.py`).

## Running benchmarks
1. **Quick smoke (mock pipeline):**
   ```bash
   python backend/benchmark_mock.py --iterations 5 --use-mock 1 --input benchmarks/01_scales/audio/c_major_scale_100bpm.wav
   ```
2. **Full pipeline for one clip:**
   ```bash
   python backend/benchmark_mock.py --iterations 5 --use-mock 0 --input benchmarks/01_scales/audio/c_major_scale_100bpm.wav
   ```
   Use this when you only want timing and MIDI/MusicXML exports without accuracy scoring.
3. **Accuracy for a single audio/reference pair:**
   ```bash
   python benchmarks/benchmark_local_file.py benchmarks/audio/Simple\ Scale\ –\ C\ Major.mp3 benchmarks/reference/c_major_scale.musicxml --use-crepe --tempo-bpm 92
   ```
   Outputs MusicXML/MIDI predictions plus a JSON summary (note counts, pitch/rhythm precision/recall/F1) in `benchmarks_results/`.
4. **Full suite accuracy snapshot:**
   ```bash
   python benchmarks/benchmark_local_file.py --suite --use-crepe --threshold 0.75 \
     --output-dir benchmarks_results \
     --report benchmarks/results_accuracy.md \
     --beat-times 0,0.48,0.96,1.44
   ```
   - `--use-crepe` enables CREPE pitch tracking when installed.
   - `--tempo-bpm` pins the quantization tempo, bypassing the 120 BPM fallback when beat tracking is unreliable.
   - `--beat-times` supplies a comma-separated beat grid (seconds) to skip beat tracking entirely; leave blank to auto-detect.
   - `--threshold` fails the command if average pitch or rhythm accuracy falls below the given fraction (default 0.75), which is useful for CI.
   - `--output-dir` collects per-clip `*_pred.musicxml`, `*_pred.mid`, and `*_summary.json` artifacts.
   - `--report` writes a Markdown rollup to `benchmarks/results_accuracy.md`.
5. **Agent mode (iterative strategies from simple → real-world):**
   ```bash
   python benchmarks/benchmark_local_file.py --suite --agent-mode --agent-target 0.9 \
     --agent-max-runs 6 --agent-tempos auto,92,100,110 --agent-prioritize-crepe
   ```
   - Starts with the simplest fixtures and tries multiple strategies (tempo overrides and pitch trackers) until both average pitch and rhythm F1 clear the target.
   - Writes per-iteration reports (if `--report` is supplied) plus `benchmarks_results/agent_history.json` so you can trace accuracy gains.

### Interpreting results
- **Pitch precision/recall/F1:** matches on MIDI pitch and onset within ±0.25 beat; false positives reduce precision and F1.
- **Rhythm precision/recall/F1:** also requires duration match within ±0.25 beat.
- **JSON summaries:** include the number of reference vs. predicted notes to quickly spot over/under-quantization.
- **MusicXML/MIDI exports:** open these alongside the reference to listen for drift or missing notes; filenames mirror the audio slug (e.g., `simple_scale_–_c_major_pred.musicxml`).

## Refining the transcription logic
The suite exercises the full pipeline. If accuracy drops, adjust these hotspots before re-running the commands above:

- **Tempo and beat tracking:** `_estimate_tempo_and_beats` in `backend/pipeline/stage_d.py` falls back to `meta.tempo_bpm` (default 120 BPM) when beat tracking fails. Tuning `hop_length` or providing better `meta` defaults can improve the beat grid.
- **Subdivision density:** `_determine_subdivisions` chooses how many grid divisions per beat (2–8) based on tempo. Increase `TARGET_SUBDIVISION_SEC` or clamp `subdivisions` to change how finely notes snap to the grid.
- **Minimum durations:** `_quantize_events` enforces a tiny positive duration when start/end quantize to the same slot. Adjust the guard or subdivision choice if you see clipped notes.
- **Measure placement and rendering:** `_build_score` inserts tempo, time signature, and key (when detected) before writing MusicXML. If measures look misaligned, verify `analysis_data.meta.time_signature` and the order of `NoteEvent` onsets.

After each change, rerun the single-clip command to inspect the exported MusicXML/MIDI, then rerun the suite to confirm pitch/rhythm accuracy improvements.

## Results template
Copy this template into each scenario's `results.md` (already seeded in the starter files):

```
# Results

| Date       | Environment (commit, OS, deps) | Fixture                          | Mode  | Iterations | Min (s) | Avg (s) | Max (s) | Notes |
|------------|--------------------------------|----------------------------------|-------|------------|---------|---------|---------|-------|
| 2025-02-15 | abc1234, Ubuntu, mock pipeline | c_major_scale_100bpm.wav         | mock  | 5          | 0.012   | 0.015   | 0.021   | Baseline sanity check |
| 2025-02-15 | abc1234, Ubuntu, full pipeline | c_major_scale_100bpm.wav         | full  | 5          | 0.480   | 0.525   | 0.610   | Initial full run |
```

### Recommended benchmark order (easy → hard)
- Very Easy: C Major Scale
- Easy: Twinkle Twinkle
- Medium: Ode to Joy
- Medium-Hard: Amazing Grace (flute, vibrato)
- Hard: Happy Birthday
