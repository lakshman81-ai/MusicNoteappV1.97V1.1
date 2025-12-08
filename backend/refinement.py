from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict

from backend.config_manager import get_config, update_config


class RefinementLoop:
    """Configuration-driven refinement orchestrator."""

    def __init__(self, metrics_dir: str | Path = "benchmarks_results") -> None:
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.best_metrics_path = self.metrics_dir / "best_metrics.json"
        self.history_path = self.metrics_dir / "config_history.json"

    def _load_metrics(self, metrics_path: Path) -> Dict[str, float]:
        if metrics_path.suffix.lower() == ".csv":
            import csv

            with open(metrics_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                return {row[0]: float(row[1]) for row in reader if len(row) >= 2}

        with open(metrics_path, "r", encoding="utf-8") as f:
            return {k: float(v) for k, v in json.load(f).items()}

    def _load_best(self) -> Dict[str, float]:
        if not self.best_metrics_path.exists():
            return {}
        with open(self.best_metrics_path, "r", encoding="utf-8") as f:
            return {k: float(v) for k, v in json.load(f).items()}

    def _save_best(self, metrics: Dict[str, float]) -> None:
        with open(self.best_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    def _record_history(self, config: Dict, metrics: Dict[str, float]) -> None:
        record = {
            "timestamp": time.time(),
            "config": config,
            "metrics": metrics,
        }
        history: list = []
        if self.history_path.exists():
            with open(self.history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        history.append(record)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _propose_updates(self, metrics: Dict[str, float]) -> Dict:
        cfg = get_config()
        updates: Dict = {}

        # Target octave stability
        if metrics.get("OA", 1.0) < 0.9:
            updates.setdefault("update", {})["fmin"] = max(40, cfg.get("fmin", 65.0) - 5)
            updates["update"]["fmax"] = max(cfg.get("fmax", 2093.0), 2000)

        # Encourage onset clarity
        if metrics.get("OnsetF", 1.0) < 0.9:
            updates.setdefault("update", {})["onset_threshold_factor"] = round(
                min(cfg.get("onset_threshold_factor", 0.25) + 0.05, 0.75), 3
            )

        # Strengthen SwiftF0 when harmonic mean lags
        if metrics.get("HM", 1.0) < 0.9:
            weights = deepcopy(cfg.get("ensemble_weights", {}))
            swift_weight = weights.get("swift", 0.4)
            yin_weight = weights.get("yin", 0.2)
            weights["swift"] = round(min(swift_weight + 0.05, 0.8), 3)
            weights["yin"] = round(max(yin_weight - 0.02, 0.05), 3)
            updates.setdefault("update", {})["ensemble_weights"] = weights

        # Prevent short spurious notes if CA is weak
        if metrics.get("CA", 1.0) < 0.9:
            updates.setdefault("update", {})["min_note_frames"] = max(3, cfg.get("min_note_frames", 3) + 1)

        return updates

    def run(self, metrics_path: str | Path, *, dry_run: bool = False) -> Dict[str, object]:
        metrics_file = Path(metrics_path)
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        metrics = self._load_metrics(metrics_file)
        best_metrics = self._load_best()
        current_config = get_config()

        updates = self._propose_updates(metrics)
        updated_config = update_config(updates.get("update", {})) if updates and not dry_run else current_config

        hm_current = metrics.get("HM", 0.0)
        hm_best = best_metrics.get("HM", 0.0)
        if hm_current >= hm_best and not dry_run:
            self._save_best(metrics)

        if not dry_run:
            self._record_history(updated_config, metrics)

        return {
            "metrics": metrics,
            "best": best_metrics,
            "applied_updates": updates.get("update", {}),
            "config": updated_config,
            "improved": hm_current >= hm_best,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configuration-driven refinement loop")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON or CSV file")
    parser.add_argument("--metrics-dir", default="benchmarks_results", help="Directory to store history and best metrics")
    parser.add_argument("--dry-run", action="store_true", help="Show proposed updates without writing config")
    args = parser.parse_args()

    loop = RefinementLoop(metrics_dir=args.metrics_dir)
    result = loop.run(args.metrics, dry_run=args.dry_run)

    print("Refinement summary")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
