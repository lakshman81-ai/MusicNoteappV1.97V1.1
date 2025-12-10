from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .common import (
    RESULTS_ROOT,
    PreparedCase,
    prepare_case_from_assets,
    prepare_polyphonic_case,
    run_iteration,
    save_phase_summary,
)

PHASE_NAME = "polyphonic_consensus"


def build_cases() -> List[PreparedCase]:
    cases: List[PreparedCase] = [
        prepare_case_from_assets(
            name="Melody plus chords",
            audio_filename="Ode To Joy.mp3",
            reference_filename="ode_to_joy.musicxml",
            scenario="polyphonic_consensus",
        ),
        prepare_polyphonic_case([48, 55, 60], duration=3.0, scenario="polyphonic_consensus/synthetic"),
    ]
    return cases


def main(target_hm: float = 0.88, min_gain: float = 0.001, max_rounds: int = 4) -> Path:
    cases = build_cases()
    output_dir = RESULTS_ROOT / PHASE_NAME

    strategies = [
        ("poly_baseline", {"use_crepe": False}),
        ("poly_crepe", {"use_crepe": True}),
        ("poly_crepe_fast_tempo", {"use_crepe": True, "tempo_override": 110.0}),
        (
            "poly_consensus_grid",
            {"use_crepe": True, "tempo_override": 110.0, "beat_times_override": [0.0, 0.75, 1.5, 2.25, 3.0]},
        ),
    ]

    history = []
    best_hm = -1.0
    exit_reason = "max_rounds"

    for idx, (strategy_name, kwargs) in enumerate(strategies, start=1):
        if idx > max_rounds:
            break
        iteration = run_iteration(cases, strategy_name=strategy_name, output_dir=output_dir, pipeline_kwargs=kwargs)
        history.append(iteration)

        hm = float(iteration.get("average_hm", 0.0))
        gain = hm - best_hm
        best_hm = max(best_hm, hm)

        if hm >= target_hm:
            exit_reason = "target_met"
            break
        if idx > 1 and gain < min_gain:
            exit_reason = "converged"
            break

    summary = {
        "phase": PHASE_NAME,
        "target_hm": target_hm,
        "min_gain": min_gain,
        "max_rounds": max_rounds,
        "exit_reason": exit_reason,
        "iterations": history,
        "best_hm": best_hm,
    }
    return save_phase_summary(PHASE_NAME, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run polyphonic consensus phase (WI-01)")
    parser.add_argument("--target-hm", type=float, default=0.88, help="Target harmonic mean threshold for exit")
    parser.add_argument("--min-gain", type=float, default=0.001, help="Minimum HM improvement required to continue tuning")
    parser.add_argument("--max-rounds", type=int, default=4, help="Maximum number of strategy rounds to attempt")
    args = parser.parse_args()

    path = main(target_hm=args.target_hm, min_gain=args.min_gain, max_rounds=args.max_rounds)
    print(f"Polyphonic consensus summary saved to: {path}")
