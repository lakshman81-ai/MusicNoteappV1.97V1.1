from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal

PitchName = Literal[
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]


# ---------- Meta / global info ----------

@dataclass
class MetaData:
    """Container for audio and score-wide metadata."""

    original_sr: Optional[int] = None
    target_sr: int = 22050
    sample_rate: int = 22050
    duration_sec: float = 0.0
    hop_length: int = 256
    time_signature: str = "4/4"
    tempo_bpm: Optional[float] = None
    detected_key: Optional[str] = None
    preprocessed_audio: Optional[Any] = None
    beat_tracking_succeeded: Optional[bool] = None

    # Legacy/auxiliary fields kept for compatibility
    tuning_offset: float = 0.0
    lufs: Optional[float] = None
    processing_mode: str = "mono"
    snr: Optional[float] = None
    window_size: int = 2048


# ---------- Pitch timeline ----------

@dataclass
class FramePitch:
    time: float
    pitch_hz: float
    midi: Optional[int]
    confidence: float


# ---------- Note events ----------

@dataclass
class AlternativePitch:
    midi: int
    confidence: float


@dataclass
class NoteEvent:
    # Raw timing
    start_sec: float
    end_sec: float

    # Pitch
    midi_note: int
    pitch_hz: float
    confidence: float = 0.0

    # Performance-ish info
    velocity: float = 0.8
    is_grace: bool = False
    dynamic: str = "mf"

    # Musical grid (filled after quantization)
    measure: Optional[int] = None
    beat: Optional[float] = None
    duration_beats: Optional[float] = None

    # Extra info
    alternatives: List[AlternativePitch] = field(default_factory=list)
    spec_thumb: Optional[str] = None


# ---------- Chords & layout ----------

@dataclass
class ChordEvent:
    time: float
    beat: float
    symbol: str
    root: str = "C"
    quality: str = "M"


@dataclass
class VexflowLayout:
    measures: List[Dict[str, Any]] = field(default_factory=list)


# ---------- All analysis data ----------

@dataclass
class AnalysisData:
    meta: MetaData = field(default_factory=MetaData)
    timeline: List[FramePitch] = field(default_factory=list)
    events: List[NoteEvent] = field(default_factory=list)
    chords: List[ChordEvent] = field(default_factory=list)
    vexflow_layout: VexflowLayout = field(default_factory=VexflowLayout)
    notes: List[NoteEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation for debugging / API."""

        return {
            "meta": {
                key: value
                for key, value in asdict(self.meta).items()
                if key != "preprocessed_audio"
            },
            "timeline": [asdict(f) for f in self.timeline],
            "notes": [
                {
                    "start_sec": e.start_sec,
                    "end_sec": e.end_sec,
                    "midi_note": e.midi_note,
                    "pitch_hz": e.pitch_hz,
                    "confidence": e.confidence,
                    "velocity": e.velocity,
                    "is_grace": e.is_grace,
                    "dynamic": e.dynamic,
                    "measure": e.measure,
                    "beat": e.beat,
                    "duration_beats": e.duration_beats,
                    "alternatives": [asdict(a) for a in e.alternatives],
                    "spec_thumb": e.spec_thumb,
                }
                for e in self.events
            ],
            "chords": [
                {
                    "time": c.time,
                    "beat": c.beat,
                    "symbol": c.symbol,
                    "root": c.root,
                    "quality": c.quality,
                }
                for c in self.chords
            ],
            "vexflow_layout": self.vexflow_layout.measures,
        }


@dataclass
class TranscriptionResult:
    musicxml: str
    analysis_data: AnalysisData
