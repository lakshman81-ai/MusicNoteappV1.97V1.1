from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from benchmarks.benchmark_local_file import (
    BenchmarkCase,
    get_default_suite,
    get_noise_robustness_cases,
    run_suite,
)

BENCHMARK_RESULTS = Path("benchmarks_results")


def _filter_cases(cases: List[BenchmarkCase], scenario: str) -> List[BenchmarkCase]:
    return [c for c in cases if c.scenario == scenario]


def _extract_metrics_from_summary(summary: Dict[str, Dict]) -> Dict[str, float]:
    # summary is {scenario: {metrics: {...}}}
    first = next(iter(summary.values()))
    return first.get("metrics", {})


def _weighted_grand_hm(summary_sets: Dict[str, Dict[str, Dict]]) -> float:
    numer = 0.0
    denom = 0.0
    for dataset_summary in summary_sets.values():
        for scenario_data in dataset_summary.values():
            case_count = float(scenario_data.get("case_count", 0))
            hm = float(scenario_data.get("metrics", {}).get("HM", 0.0))
            numer += hm * case_count
            denom += case_count
    return numer / denom if denom else 0.0


def main() -> None:
    base_cases = get_default_suite()
    datasets = {
        "01_scales": _filter_cases(base_cases, "01_scales"),
        "02_simple_melodies": _filter_cases(base_cases, "02_simple_melodies"),
        "03_melody_plus_chords": _filter_cases(base_cases, "03_melody_plus_chords"),
        "04_pop_loops": _filter_cases(base_cases, "04_pop_loops"),
    }

    results: Dict[str, Dict] = {}
    for name, cases in datasets.items():
        suite_result = run_suite(cases, output_dir=BENCHMARK_RESULTS)
        results[name] = suite_result.summary

    noise_base_cases = datasets["01_scales"] + datasets["02_simple_melodies"] + datasets["03_melody_plus_chords"]
    noise_suites = get_noise_robustness_cases(noise_base_cases)
    noise_results: Dict[str, Dict] = {}
    for level, cases in noise_suites.items():
        suite_result = run_suite(cases, output_dir=BENCHMARK_RESULTS)
        noise_results[level] = suite_result.summary

    datasets_metrics: Dict[str, Dict] = {}
    for dataset_name, suite_summary in results.items():
        datasets_metrics[dataset_name] = _extract_metrics_from_summary(suite_summary)

    noise_metrics: Dict[str, Dict] = {}
    for level, summary in noise_results.items():
        noise_metrics[level] = _extract_metrics_from_summary(summary)

    global_summary = {
        "datasets": {
            **datasets_metrics,
            "05_noise_robustness": noise_metrics,
        },
        "grand_HM": _weighted_grand_hm({**results, **noise_results}),
    }

    BENCHMARK_RESULTS.mkdir(parents=True, exist_ok=True)
    output_path = BENCHMARK_RESULTS / "global_summary.json"
    output_path.write_text(json.dumps(global_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Global summary saved to: {output_path}")


if __name__ == "__main__":
    main()
