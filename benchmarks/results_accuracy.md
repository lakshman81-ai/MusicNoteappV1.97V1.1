# Benchmark accuracy snapshot

Run `python benchmarks/benchmark_local_file.py --suite` to regenerate this table and the JSON summaries in `benchmarks_results/`. The suite spans the numbered benchmark scenarios with paired references so we can compute pitch and rhythm accuracy in one pass.

| Scenario | Audio | Reference | Pitch accuracy | Rhythm accuracy (±0.25 beat) | Notes |
|----------|-------|-----------|----------------|-------------------------------|-------|
| 01_scales | Simple Scale – C Major.mp3 | benchmarks/reference/c_major_scale.musicxml | Pending | Pending | Regenerate via `--suite` to capture the latest pipeline output. |
| 02_simple_melodies | Twinkle_Twinkle_Little_Star.mp3 | benchmarks/reference/twinkle_twinkle.musicxml | Pending | Pending | Regenerate via `--suite` to capture the latest pipeline output. |
| 02_simple_melodies | Happy_birthday_to_you.mp3 | benchmarks/reference/HappyBirthday.mid | Pending | Pending | Regenerate via `--suite` to capture the latest pipeline output. |
| 03_melody_plus_chords | Ode To Joy.mp3 | benchmarks/reference/ode_to_joy.musicxml | Pending | Pending | Regenerate via `--suite` to capture the latest pipeline output. |
| 04_pop_loops | AmazingGraceFamiliarStyle.mp3 | benchmarks/reference/amazing_grace.musicxml | Pending | Pending | Regenerate via `--suite` to capture the latest pipeline output. |

## Averages
- Pitch accuracy: Pending
- Rhythm accuracy: Pending
- Threshold: 75%
