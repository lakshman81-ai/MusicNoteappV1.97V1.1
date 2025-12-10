from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .common import (
    RESULTS_ROOT,
    PreparedCase,
    prepare_case_from_assets,
    prepare_noisy_case,
    run_iteration,
    save_phase_summary,
)

PHASE_NAME = "noise_stress"
DEFAULT_SNR_LEVELS = (20, 10)


def build_base_cases() -> List[PreparedCase]:
    return [
        prepare_case_from_assets(
            name="Ode To Joy",
            audio_filename="Ode To Joy.mp3",
            reference_filename="ode_to_joy.musicxml",
            scenario="noise_robustness",
        ),
        prepare_case_from_assets(
            name="Amazing Grace (Familiar Style)",
            audio_filename="AmazingGraceFamiliarStyle.mp3",
            reference_filename="amazing_grace.musicxml",
            scenario="noise_robustness",
        ),
    ]


def build_noisy_cases(snr_levels: List[int]) -> Dict[int, List[PreparedCase]]:
    base_cases = build_base_cases()
    suites: Dict[int, List[PreparedCase]] = {}
    for snr in snr_levels:
        suites[snr] = [prepare_noisy_case(case, snr_db=snr) for case in base_cases]
    return suites


def main(target_hm: float = 0.82, min_gain: float = 0.0015, max_rounds: int = 3, snr_levels: List[int] | None = None) -> Path:
    snr_levels = snr_levels or list(DEFAULT_SNR_LEVELS)
    suites = build_noisy_cases(snr_levels)
    output_dir = RESULTS_ROOT / PHASE_NAME

    strategies = [
        ("noise_baseline", {"use_crepe": False}),
        ("noise_crepe", {"use_crepe": True}),
        (
            "noise_crepe_beat_grid",
            {"use_crepe": True, "beat_times_override": [0.0, 0.5, 1.0, 1.5, 2.0]},
        ),
    ]

    results_by_snr: Dict[int, Dict[str, object]] = {}

    for snr in snr_levels:
        history = []
        best_hm = -1.0
        exit_reason = "max_rounds"
        cases = suites[snr]

        for idx, (strategy_name, kwargs) in enumerate(strategies, start=1):
            if idx > max_rounds:
                break
            iteration = run_iteration(
                cases,
                strategy_name=strategy_name,
                output_dir=output_dir / f"SNR{snr}",
                pipeline_kwargs=kwargs,
            )
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

        results_by_snr[snr] = {
            "target_hm": target_hm,
            "min_gain": min_gain,
            "max_rounds": max_rounds,
            "exit_reason": exit_reason,
            "iterations": history,
            "best_hm": best_hm,
        }

    summary = {"phase": PHASE_NAME, "results": results_by_snr}
    return save_phase_summary(PHASE_NAME, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run noise stress phase at SNR 20/10 dB (WI-01)")
    parser.add_argument("--target-hm", type=float, default=0.82, help="Target harmonic mean threshold for exit")
    parser.add_argument("--min-gain", type=float, default=0.0015, help="Minimum HM improvement required to continue tuning")
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum number of strategy rounds to attempt")
    parser.add_argument(
        "--snr-levels",
        type=int,
        nargs="*",
        default=list(DEFAULT_SNR_LEVELS),
        help="Noise stress SNR levels to exercise",
    )
    args = parser.parse_args()

    path = main(target_hm=args.target_hm, min_gain=args.min_gain, max_rounds=args.max_rounds, snr_levels=args.snr_levels)
    print(f"Noise stress summaries saved to: {path}")
