# Benchmark accuracy snapshot

Current pitch and rhythm accuracy measured with `benchmarks/benchmark_local_file.py`.

| Audio | Reference | Pitch accuracy | Rhythm accuracy (±0.25 beat) | Notes |
|-------|-----------|----------------|-------------------------------|-------|
| Simple Scale – C Major | benchmarks/reference/c_major_scale.musicxml | 6.7% | 0.0% | 14 predicted vs 15 reference notes; persistent octave and duration errors. |
| Happy_birthday_to_you | benchmarks/reference/HappyBirthday.mid | 8.8% | 0.7% | 203 predicted vs 137 reference notes; over-segmentation and timing drift remain. |

