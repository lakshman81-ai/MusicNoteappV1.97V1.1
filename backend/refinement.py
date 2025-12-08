from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

from backend.config_manager import get_config, update_config


class RefinementLoop:
    """Configuration-driven refinement orchestrator based on metric suites."""

    def __init__(self, metrics_dir: str | Path = "benchmarks_results") -> None:
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.metrics_dir / "refinement_history.jsonl"

    def _load_metrics(self, metrics_path: Path) -> Dict:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _append_history(self, record: Dict) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _flatten_metrics(self, metrics: Dict) -> Dict[str, Dict[str, float]]:
        if "datasets" in metrics:
            return {k: v for k, v in metrics["datasets"].items() if isinstance(v, dict)}
        if "metrics" in metrics:
            return {"primary": metrics["metrics"]}
        if any(isinstance(v, (int, float)) for v in metrics.values()):
            return {"primary": metrics}
        return {}

    def _aggregate_metrics(self, metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
        aggregate: Dict[str, float] = {}
        metrics_list = list(metrics)
        if not metrics_list:
            return aggregate
        keys = {k for m in metrics_list for k in m.keys()}
        for key in keys:
            aggregate[key] = sum(float(m.get(key, 0.0)) for m in metrics_list) / len(metrics_list)
        return aggregate

    def _noise_baseline_gap(self, datasets: Dict[str, Dict[str, float]]) -> float:
        noise = datasets.get("05_noise_robustness")
        if not isinstance(noise, dict):
            return 0.0
        hm_values = {}
        for level, vals in noise.items():
            if isinstance(vals, dict) and "HM" in vals:
                hm_values[level] = float(vals.get("HM", 0.0))
        if not hm_values:
            return 0.0
        baseline = hm_values.get("SNR20") or max(hm_values.values())
        worst = min(hm_values.values())
        return baseline - worst

    def _apply_rules(self, metrics: Dict[str, float], datasets: Dict[str, Dict[str, float]]) -> Dict:
        cfg = get_config()
        updates: Dict = {}

        target_hm = float(cfg.get("target_hm", 0.9))
        target_onset_f = float(cfg.get("target_onset_f", 0.9))

        current_weights = cfg.get("ensemble_weights", {})
        swift_weight = float(current_weights.get("swift", 0.4))
        yin_weight = float(current_weights.get("yin", 0.2))

        hm = float(metrics.get("HM", 0.0))
        onset_f = float(metrics.get("OnsetF", 0.0))
        ca = float(metrics.get("CA", 0.0))
        oa = float(metrics.get("OA", 0.0))

        if hm < target_hm:
            new_swift = min(swift_weight + 0.05, 1.0)
            new_yin = max(yin_weight - 0.05, 0.0)
            updates.setdefault("ensemble_weights", {})["swift"] = round(new_swift, 3)
            updates.setdefault("ensemble_weights", {})["yin"] = round(new_yin, 3)
            updates["confidence_floor"] = round(float(cfg.get("confidence_floor", 0.1)) + 0.02, 3)
            updates["median_window"] = int(cfg.get("median_window", 11)) + 2

        if onset_f < target_onset_f:
            updates["onset_threshold_factor"] = round(float(cfg.get("onset_threshold_factor", 0.25)) + 0.05, 3)
            updates["min_note_frames"] = int(cfg.get("min_note_frames", 3)) + 1

        if ca < target_hm or oa < target_hm:
            updates["min_note_frames"] = int(cfg.get("min_note_frames", 3)) + 1
            updates["fmax"] = round(float(cfg.get("fmax", 2093.0)) * 0.9, 2)
            updates["fmin"] = max(float(cfg.get("fmin", 65.0)), 80.0)

        noise_gap = self._noise_baseline_gap(datasets)
        if noise_gap > 0.05:
            updates["median_window"] = int(updates.get("median_window", cfg.get("median_window", 11))) + 2
            updates["confidence_floor"] = round(float(updates.get("confidence_floor", cfg.get("confidence_floor", 0.1))) + 0.02, 3)
            updates.setdefault("ensemble_weights", {})["swift"] = round(updates.get("ensemble_weights", {}).get("swift", swift_weight) + 0.05, 3)

        return updates

    def run(self, metrics_path: str | Path, *, dry_run: bool = False) -> Dict[str, object]:
        metrics_file = Path(metrics_path)
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        raw_metrics = self._load_metrics(metrics_file)
        datasets = self._flatten_metrics(raw_metrics)
        aggregate_metrics = self._aggregate_metrics(datasets.values()) if datasets else raw_metrics.get(
            "metrics", raw_metrics
        )

        updates = self._apply_rules(aggregate_metrics, datasets)
        config_before = get_config()
        config_after = config_before if dry_run else update_config(updates)

        record = {
            "timestamp": time.time(),
            "changes": updates,
            "metrics_before": aggregate_metrics,
            "metrics_after": aggregate_metrics,
            "config_after": config_after,
            "metrics_source": str(metrics_file),
        }
        if not dry_run:
            self._append_history(record)

        return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configuration-driven refinement loop")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON file")
    parser.add_argument("--metrics-dir", default="benchmarks_results", help="Directory to store history")
    parser.add_argument("--dry-run", action="store_true", help="Show proposed updates without writing config")
    args = parser.parse_args()

    loop = RefinementLoop(metrics_dir=args.metrics_dir)
    result = loop.run(args.metrics, dry_run=args.dry_run)

    print("Refinement summary")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
