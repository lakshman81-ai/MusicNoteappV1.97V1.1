import unittest
from unittest.mock import patch

import numpy as np

from backend.pipeline import stage_b
from backend.pipeline.models import MetaData


class ExtractFeaturesCrepeToggleTests(unittest.TestCase):
    def test_crepe_is_skipped_when_flag_is_false(self):
        meta = MetaData()
        waveform = np.zeros(1024, dtype=np.float32)

        with patch("backend.pipeline.stage_b._crepe_available", return_value=True), patch(
            "backend.pipeline.stage_b._pitch_with_crepe"
        ) as crepe_mock, patch(
            "backend.pipeline.stage_b._pitch_with_pyin",
            return_value=(
                np.asarray([0.0]),
                np.asarray([440.0]),
                np.asarray([True]),
                np.asarray([0.9]),
            ),
        ) as pyin_mock, patch(
            "backend.pipeline.stage_b._estimate_polyphonic_peaks", return_value=[]
        ), patch(
            "backend.pipeline.stage_b._harmonic_summation_refine", side_effect=lambda *_args, **_kwargs: np.asarray([440.0])
        ), patch(
            "backend.pipeline.stage_b._smooth_midi_with_voicing", side_effect=lambda f0, _probs: f0
        ), patch(
            "backend.pipeline.stage_b._build_timeline",
            return_value=[stage_b.FramePitch(time=0.0, pitch_hz=440.0, midi=69, confidence=0.9)],
        ), patch("backend.pipeline.stage_b._segment_notes_from_timeline", return_value=[]):
            timeline, notes, chords = stage_b.extract_features(
                waveform, 22050, meta, use_crepe=False
            )

        self.assertTrue(pyin_mock.called)
        self.assertFalse(crepe_mock.called)
        self.assertEqual(len(timeline), 1)
        self.assertEqual(notes, [])
        self.assertEqual(chords, [])


if __name__ == "__main__":
    unittest.main()
