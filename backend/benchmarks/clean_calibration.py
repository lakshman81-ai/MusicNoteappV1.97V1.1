from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .common import (
    RESULTS_ROOT,
    PreparedCase,
    prepare_case_from_assets,
    run_iteration,
    save_phase_summary,
)

PHASE_NAME = "clean_calibration"


def build_cases() -> List[PreparedCase]:
    return [
        prepare_case_from_assets(
            name="Simple Scale – C Major",
            audio_filename="Simple Scale – C Major.mp3",
            reference_filename="c_major_scale.musicxml",
            scenario="01_scales",
        ),
        prepare_case_from_assets(
            name="Twinkle Twinkle Little Star",
            audio_filename="Twinkle_Twinkle_Little_Star.mp3",
            reference_filename="twinkle_twinkle.musicxml",
            scenario="02_simple_melodies",
        ),
    ]


def main(target_hm: float = 0.9, min_gain: float = 0.002, max_rounds: int = 3) -> Path:
    cases = build_cases()
    output_dir = RESULTS_ROOT / PHASE_NAME

    strategies = [
        ("baseline", {"use_crepe": False}),
        ("crepe_enabled", {"use_crepe": True}),
        ("crepe_tempo_guided", {"use_crepe": True, "tempo_override": 96.0}),
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
    parser = argparse.ArgumentParser(description="Run clean calibration benchmark phase (WI-01)")
    parser.add_argument("--target-hm", type=float, default=0.9, help="Target harmonic mean threshold for exit")
    parser.add_argument("--min-gain", type=float, default=0.002, help="Minimum HM improvement required to continue tuning")
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum number of strategy rounds to attempt")
    args = parser.parse_args()

    path = main(target_hm=args.target_hm, min_gain=args.min_gain, max_rounds=args.max_rounds)
    print(f"Clean calibration summary saved to: {path}")
