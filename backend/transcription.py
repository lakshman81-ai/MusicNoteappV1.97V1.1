from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable

from pipeline.models import AnalysisData, TranscriptionResult, MetaData


def _load_pipeline_stages() -> tuple[
    Callable,
    Callable,
    Callable,
    Callable,
]:
    """
    Lazily import heavy pipeline dependencies so mock-mode can run
    without requiring audio libraries to be installed.
    """

    from pipeline.stage_a import load_and_preprocess
    from pipeline.stage_b import extract_features
    from pipeline.stage_c import apply_theory
    from pipeline.stage_d import quantize_and_render

    return (
        load_and_preprocess,
        extract_features,
        apply_theory,
        quantize_and_render,
    )


def transcribe_audio_pipeline(
    file_path: str,
    stereo_mode: bool = False,
    use_mock: bool = False,
    start_offset: float = 0.0,
    max_duration: float | None = None,
) -> TranscriptionResult:
    """
    Main orchestrator for the transcription pipeline.

    Pipeline:
        Stage A: load_and_preprocess
        Stage B: extract_features (timeline + notes)
        Stage C: apply_theory (dynamics, grace, key)
        Stage D: quantize_and_render (MusicXML)
    """
    if use_mock:
        from pathlib import Path

        xml_path = Path(__file__).parent / "mock_data" / "happy_birthday.xml"
        musicxml = xml_path.read_text(encoding="utf-8")
        analysis_data = AnalysisData(meta=MetaData())
        return TranscriptionResult(musicxml=musicxml, analysis_data=analysis_data)

    (
        load_and_preprocess,
        extract_features,
        apply_theory,
        quantize_and_render,
    ) = _load_pipeline_stages()

    tmp_dir = tempfile.mkdtemp(prefix="mnc_")
    try:
        tmp_file = os.path.join(tmp_dir, os.path.basename(file_path))
        shutil.copy2(file_path, tmp_file)

        # Stage A
        y, sr, meta = load_and_preprocess(
            tmp_file,
            stereo_mode=stereo_mode,
            start_offset=start_offset,
            max_duration=max_duration,
        )

        analysis_data = AnalysisData(meta=meta)

        # Stage B
        timeline, notes, chords = extract_features(y, sr, meta=meta)
        analysis_data.timeline = timeline
        analysis_data.chords = chords

        # Stage C
        events_with_theory = apply_theory(notes, analysis_data)

        # Stage D
        musicxml = quantize_and_render(events_with_theory, analysis_data)

        return TranscriptionResult(musicxml=musicxml, analysis_data=analysis_data)

    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


def transcribe_audio(
    file_path: str,
    use_mock: bool = False,
    stereo_mode: bool = False,
) -> str:
    """
    Legacy entry point, returns just the MusicXML string.
    """
    result = transcribe_audio_pipeline(
        file_path,
        stereo_mode=stereo_mode,
        use_mock=use_mock,
    )
    return result.musicxml
