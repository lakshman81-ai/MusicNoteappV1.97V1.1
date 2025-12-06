import argparse
import json
import logging
import os
import sys
import numpy as np
import librosa
import music21

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio to music notation")
    parser.add_argument("--audio_path", required=True, help="Path to input audio file")
    parser.add_argument("--audio_start_offset_sec", type=float, default=10.0, help="Start offset in seconds")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate")
    parser.add_argument("--output_musicxml", default="output.musicxml", help="Output MusicXML path")
    parser.add_argument("--output_midi", default="output.mid", help="Output MIDI path")
    parser.add_argument("--output_png", default="output.png", help="Output PNG path")
    parser.add_argument("--output_log", default="transcription_log.json", help="Output log path")
    return parser.parse_args()

def freq_to_midi(freq):
    if freq is None or freq <= 0:
        return None
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def quantize_duration(seconds, bpm, denominators=[4.0, 2.0, 1.0, 0.5, 0.25, 0.125]):
    # beats = seconds * (bpm / 60)
    beats = seconds * (bpm / 60.0)

    # Find nearest denominator
    closest = min(denominators, key=lambda x: abs(x - beats))

    # Simple quantization
    return closest, beats

def main():
    args = parse_args()

    # Parameters from WI
    params = {
        "f0_fmin": librosa.note_to_hz('C2'),
        "f0_fmax": librosa.note_to_hz('C6'),
        "frame_length": 2048,
        "hop_length": 512,
        "pitch_smoothing_ms": 75,
        "min_note_duration_sec": 0.06,
        "merge_gap_threshold_sec": 0.15,
        "quantization_tolerance": 0.20,
        "rhythmic_denominators": [4.0, 2.0, 1.0, 0.5, 0.25, 0.125],
        "split_midi_threshold": 60
    }

    logger.info(f"Starting transcription for {args.audio_path}")

    if not os.path.exists(args.audio_path):
        logger.error(f"Audio file not found: {args.audio_path}")
        sys.exit(1)

    # 1. Load Audio
    logger.info("Step 1: Loading Audio...")
    try:
        y, sr = librosa.load(args.audio_path, sr=args.sample_rate, offset=args.audio_start_offset_sec)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        sys.exit(1)

    # 2. Preprocess (Normalization)
    logger.info("Step 2: Preprocessing...")
    y = librosa.util.normalize(y)

    # 3. Tempo and Beats
    logger.info("Step 3: Estimating Tempo...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    logger.info(f"Detected tempo: {tempo} BPM")

    # 4. Pitch Tracking (pyin)
    logger.info("Step 4: Pitch Tracking...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=params["f0_fmin"],
        fmax=params["f0_fmax"],
        sr=sr,
        frame_length=params["frame_length"],
        hop_length=params["hop_length"]
    )

    times = librosa.times_like(f0, sr=sr, hop_length=params["hop_length"])

    # 7. Segment Notes (Simplified logic merging Steps 5-8)
    logger.info("Step 7: Segmenting Notes...")

    current_midi = None
    start_time = None

    # Convert f0 sequence to MIDI sequence (handling None/unvoiced)
    midi_sequence = [freq_to_midi(f) if v else None for f, v in zip(f0, voiced_flag)]

    note_events = []

    for i, midi in enumerate(midi_sequence):
        t = times[i]

        if midi is None:
            if current_midi is not None:
                # End note
                duration = t - start_time
                if duration >= params["min_note_duration_sec"]:
                    note_events.append({
                        "start_sec": start_time,
                        "end_sec": t,
                        "midi": current_midi
                    })
                current_midi = None
                start_time = None
        else:
            if current_midi is None:
                # Start note
                current_midi = midi
                start_time = t
            elif midi != current_midi:
                # Pitch change -> End current, start new
                duration = t - start_time
                if duration >= params["min_note_duration_sec"]:
                     note_events.append({
                        "start_sec": start_time,
                        "end_sec": t,
                        "midi": current_midi
                    })
                current_midi = midi
                start_time = t

    # Close last note if active
    if current_midi is not None:
         note_events.append({
            "start_sec": start_time,
            "end_sec": times[-1],
            "midi": current_midi
        })

    # 8. Merge Adjacent Same-Pitch Notes
    logger.info("Step 8: Merging Notes...")
    merged_notes = []
    if note_events:
        merged_notes.append(note_events[0])
        for n in note_events[1:]:
            last = merged_notes[-1]
            gap = n["start_sec"] - last["end_sec"]
            if n["midi"] == last["midi"] and gap < params["merge_gap_threshold_sec"]:
                last["end_sec"] = n["end_sec"]
            else:
                merged_notes.append(n)

    logger.info(f"Total notes extracted: {len(merged_notes)}")

    # 9-14. Quantization, Voice Assignment, MusicXML
    logger.info("Step 9-14: Building Score...")

    s = music21.stream.Score()
    p_treble = music21.stream.Part()
    p_bass = music21.stream.Part()

    p_treble.insert(0, music21.clef.TrebleClef())
    p_bass.insert(0, music21.clef.BassClef())

    # Tempo
    mm = music21.tempo.MetronomeMark(number=tempo)
    p_treble.insert(0, mm)

    log_entries = []

    for n in merged_notes:
        dur_sec = n["end_sec"] - n["start_sec"]
        q_dur, q_beats = quantize_duration(dur_sec, tempo, params["rhythmic_denominators"])

        m21_note = music21.note.Note(n["midi"])
        m21_note.quarterLength = q_beats

        # Calculate start beat
        start_beat = n["start_sec"] * (tempo / 60.0)

        # Determine staff
        if n["midi"] >= params["split_midi_threshold"]:
            p_treble.insert(start_beat, m21_note)
            staff = "treble"
        else:
            p_bass.insert(start_beat, m21_note)
            staff = "bass"

        log_entries.append({
            "start_sec": n["start_sec"],
            "end_sec": n["end_sec"],
            "midi": n["midi"],
            "quantized_rhythm": q_beats,
            "start_beat": start_beat,
            "staff": staff
        })

    s.insert(0, p_treble)
    s.insert(0, p_bass)

    # 13. Key Detection
    try:
        key = s.analyze('key')
        p_treble.insert(0, key)
        logger.info(f"Detected key: {key}")
    except Exception as e:
        logger.warning(f"Key detection failed: {e}")

    # Make Measures and Ties (Crucial for notation)
    logger.info("Structuring measures and ties...")
    try:
        s.makeMeasures(inPlace=True)
        s.makeTies(inPlace=True)
    except Exception as e:
        logger.error(f"Failed to make measures/ties: {e}")

    # 14. Render Output
    logger.info("Step 14: Writing Output Files...")
    try:
        s.write('musicxml', fp=args.output_musicxml)
        logger.info(f"Written MusicXML to {args.output_musicxml}")
    except Exception as e:
        logger.error(f"Failed to write MusicXML: {e}")

    try:
        s.write('midi', fp=args.output_midi)
        logger.info(f"Written MIDI to {args.output_midi}")
    except Exception as e:
        logger.error(f"Failed to write MIDI: {e}")

    # 15. Render PNG
    try:
        # Attempts to use external helper (MuseScore/LilyPond)
        s.write('musicxml.png', fp=args.output_png)
        logger.info(f"Written PNG to {args.output_png}")
    except Exception as e:
        logger.warning(f"PNG generation failed (environment dependencies likely missing): {e}")

    # 16. Logging
    with open(args.output_log, 'w') as f:
        json.dump(log_entries, f, indent=2)
    logger.info(f"Written log to {args.output_log}")

if __name__ == "__main__":
    main()
