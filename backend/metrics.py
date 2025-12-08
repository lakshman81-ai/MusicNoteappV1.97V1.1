from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from backend.config_manager import get_config_value


@dataclass
class NoteMatch:
    reference_idx: int
    predicted_idx: int
    onset_diff: float
    offset_diff: float
    pitch_diff_cents: float


def _note_to_tuple(note) -> Tuple[int, float, float]:
    return int(note.midi_note), float(note.start_sec), float(note.end_sec)


def _pitch_diff_cents(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("inf")
    return abs(1200.0 * np.log2(a / b))


def _match_notes(reference: Iterable, predicted: Iterable, onset_tol: float, offset_tol: float, pitch_tol_cents: float) -> List[NoteMatch]:
    ref_list = list(reference)
    pred_list = list(predicted)
    matches: List[NoteMatch] = []
    used_pred = set()

    for ref_idx, ref in enumerate(ref_list):
        ref_midi, ref_on, ref_off = _note_to_tuple(ref)
        best_idx = None
        best_score = None
        for pred_idx, pred in enumerate(pred_list):
            if pred_idx in used_pred:
                continue
            midi, on, off = _note_to_tuple(pred)
            pitch_diff = abs(ref_midi - midi)
            onset_diff = abs(ref_on - on)
            offset_diff = abs(ref_off - off)
            cents = _pitch_diff_cents(pred.pitch_hz, ref.pitch_hz)
            if pitch_diff > 0 and cents > pitch_tol_cents:
                continue
            if onset_diff > onset_tol or offset_diff > offset_tol:
                continue
            score = onset_diff + offset_diff + cents / 1200.0
            if best_score is None or score < best_score:
                best_idx = pred_idx
                best_score = score
                best_match = NoteMatch(ref_idx, pred_idx, onset_diff, offset_diff, cents)
        if best_idx is not None:
            matches.append(best_match)
            used_pred.add(best_idx)
    return matches


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_voicing_scores(reference: List, predicted: List, matches: List[NoteMatch]) -> Tuple[float, float]:
    voiced_ref = len(reference)
    voiced_pred = len(predicted)
    matched = len(matches)
    precision = matched / voiced_pred if voiced_pred else 0.0
    recall = matched / voiced_ref if voiced_ref else 0.0
    return precision, recall


def compute_metrics(reference_notes: List, predicted_notes: List) -> Dict[str, float]:
    onset_tol = float(get_config_value("onset_tolerance_sec", 0.05))
    offset_tol = float(get_config_value("offset_tolerance_sec", 0.08))
    pitch_tol = float(get_config_value("pitch_tolerance_cents", 50.0))
    gross_error_cents = float(get_config_value("gross_error_cents", 100.0))

    matches = _match_notes(reference_notes, predicted_notes, onset_tol, offset_tol, pitch_tol)

    precision, recall = compute_voicing_scores(reference_notes, predicted_notes, matches)
    onset_precision = precision
    onset_recall = recall
    onset_f1 = _f1(onset_precision, onset_recall)

    offset_hits = [m for m in matches if m.offset_diff <= offset_tol]
    offset_precision = len(offset_hits) / len(predicted_notes) if predicted_notes else 0.0
    offset_recall = len(offset_hits) / len(reference_notes) if reference_notes else 0.0
    onset_offset_f1 = _f1(offset_precision, offset_recall)

    rpa = len(matches) / len(reference_notes) if reference_notes else 0.0
    chromatic_hits = [m for m in matches if abs(m.pitch_diff_cents) <= pitch_tol]
    ca = len(chromatic_hits) / len(reference_notes) if reference_notes else 0.0
    octave_hits = [m for m in matches if (abs(_note_to_tuple(reference_notes[m.reference_idx])[0] - _note_to_tuple(predicted_notes[m.predicted_idx])[0]) % 12) == 0]
    oa = len(octave_hits) / len(reference_notes) if reference_notes else 0.0
    gross_errors = [m for m in matches if m.pitch_diff_cents > gross_error_cents]
    gea = 1.0 - (len(gross_errors) / len(reference_notes) if reference_notes else 0.0)

    hm = _f1(rpa, onset_f1)

    return {
        "RPA": rpa,
        "CA": ca,
        "OA": oa,
        "GEA": gea,
        "HM": hm,
        "OnsetPrecision": onset_precision,
        "OnsetRecall": onset_recall,
        "OnsetF": onset_f1,
        "OffsetPrecision": offset_precision,
        "OffsetRecall": offset_recall,
        "OnsetOffsetF": onset_offset_f1,
        "VoicingPrecision": precision,
        "VoicingRecall": recall,
    }


def save_metrics(metrics: Dict[str, float], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def save_metrics_csv(metrics: Dict[str, float], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
