import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.transcription import transcribe_audio_pipeline


class TranscriptionPipelineTests(unittest.TestCase):
    def setUp(self):
        sr = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, tone, sr)
        self.audio_path = tmp.name

    def tearDown(self):
        Path(self.audio_path).unlink(missing_ok=True)

    def test_pipeline_returns_expected_keys(self):
        result = transcribe_audio_pipeline(self.audio_path)
        self.assertIn("musicxml", result)
        self.assertIn("midi_bytes", result)
        self.assertIn("meta", result)
        self.assertIn("notes", result)
        self.assertIsInstance(result["musicxml"], str)
        self.assertIsInstance(result["midi_bytes"], (bytes, bytearray))


if __name__ == "__main__":
    unittest.main()
