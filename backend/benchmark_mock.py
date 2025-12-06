"""
Simple benchmarking harness for the transcription pipeline.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.transcription import transcribe_audio_pipeline


def run_benchmark(iterations: int, input_path: str) -> None:
    target_path = Path(input_path)

    durations: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = transcribe_audio_pipeline(str(target_path))
        durations.append(time.perf_counter() - start)

    print("Benchmark results")
    print(f"  Iterations: {iterations}")
    print(f"  Input: {target_path}")
    print(f"  Output length: {len(result['musicxml'])} characters")
    print(f"  Min / Avg / Max: {min(durations):.4f}s / {statistics.mean(durations):.4f}s / {max(durations):.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the transcription pipeline")
    parser.add_argument("--iterations", type=int, default=5, help="Number of runs to average")
    parser.add_argument("--input", type=str, required=True, help="Audio file path")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, input_path=args.input)
