from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal


PitchName = Literal[
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


# ---------- Meta / global info ----------

@dataclass
class MetaData:
    tuning_offset: float = 0.0          # in semitones (fractional)
    detected_key: str = "C"            # e.g. "C", "Gm"
    lufs: float = -14.0                # integrated loudness in LUFS
    processing_mode: str = "mono"      # "mono" | "stereo" | "polyphonic"
    snr: float = 0.0                   # signal-to-noise estimate
    window_size: int = 2048            # analysis window size
    hop_length: int = 512              # analysis hop length
    sample_rate: int = 22050           # analysis sample rate
    tempo_bpm: float = 120.0           # global tempo estimate
    time_signature: str = "4/4"        # default, can be improved later


# ---------- Pitch timeline ----------

@dataclass
class FramePitch:
    time: float                        # seconds
    pitch_hz: float                    # 0.0 if unvoiced
    midi: Optional[int]                # None if unvoiced
    confidence: float                  # 0–1


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
    velocity: float = 0.8              # 0–1 (later mapped to MIDI 0–127)
    is_grace: bool = False
    dynamic: str = "mf"                # "p", "mf", "f", etc.

    # Musical grid (filled after quantization)
    measure: Optional[int] = None
    beat: Optional[float] = None       # beat in measure (1.0, 1.5, etc.)
    duration_beats: Optional[float] = None

    # Extra info
    alternatives: List[AlternativePitch] = field(default_factory=list)
    spec_thumb: Optional[str] = None   # optional spectrogram thumbnail id


# ---------- Chords & layout ----------

@dataclass
class ChordEvent:
    time: float                        # seconds
    beat: float                        # global beat index
    symbol: str                        # e.g. "C", "G7", "Am"
    root: str = "C"
    quality: str = "M"                 # "M", "m", "7", etc.


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

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-serializable representation for debugging / API.
        """
        return {
            "meta": asdict(self.meta),
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
