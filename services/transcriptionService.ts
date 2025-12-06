
import { audioEngine } from './audioEngine';
import { NoteEvent, ChordEvent, AnalysisDataLite, TranscriptionMeta } from '../types';
import { detectChords } from '../utils/chordDetection';

// Helper to format Harmony XML
const getMusicXMLHarmony = (chord: ChordEvent): string => {
    let rootStep = chord.root.charAt(0);
    let rootAlter = 0;
    
    if (chord.root.includes('#')) rootAlter = 1;
    if (chord.root.includes('b')) rootAlter = -1;

    // Map quality to MusicXML kind
    let kind = 'major';
    if (chord.quality === 'm') kind = 'minor';
    if (chord.quality === 'dim') kind = 'diminished';
    if (chord.quality === 'aug') kind = 'augmented';
    if (chord.quality === '7') kind = 'dominant';
    if (chord.quality === 'maj7') kind = 'major-seventh';
    if (chord.quality === 'm7') kind = 'minor-seventh';
    if (chord.quality === 'sus4') kind = 'suspended-fourth';
    if (chord.quality === 'sus2') kind = 'suspended-second';

    let alterTag = '';
    if (rootAlter !== 0) {
        alterTag = `<root-alter>${rootAlter}</root-alter>`;
    }

    return `
      <harmony>
        <root>
          <root-step>${rootStep}</root-step>
          ${alterTag}
        </root>
        <kind>${kind}</kind>
      </harmony>`;
};

// Simple MusicXML Generator for Client-Side Fallback
const createMusicXML = (
    notes: NoteEvent[],
    chords: ChordEvent[],
    options: { tempoBpm?: number; timeSignature?: string } = {}
): string => {
    const tempoBpm = options.tempoBpm ?? 120;
    const timeSignature = options.timeSignature ?? '4/4';
    const [beatsPerMeasure, beatType] = timeSignature.split('/').map(Number);
    const secondsPerBeat = 60 / tempoBpm;
    const DIVISIONS = 24;

    const sanitizedNotes = notes
        .map(n => {
            const startBeat = n.startBeat ?? n.start_time / secondsPerBeat;
            const durationBeats = Math.max(n.durationBeats ?? n.duration / secondsPerBeat, 1 / DIVISIONS);
            return { ...n, startBeat, durationBeats };
        })
        .sort((a, b) => a.startBeat - b.startBeat || a.midi_pitch - b.midi_pitch);

    const chordBeats = chords
        .map(c => ({ ...c, startBeat: c.startBeat ?? c.beat ?? c.time / secondsPerBeat }))
        .sort((a, b) => (a.startBeat ?? 0) - (b.startBeat ?? 0));

    let xml = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <work><work-title>Analyzed Audio</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">`;

    if (sanitizedNotes.length === 0) {
        return (
            xml +
            `<measure number="1"><attributes><divisions>${DIVISIONS}</divisions><time><beats>${beatsPerMeasure}</beats><beat-type>${beatType}</beat-type></time></attributes><direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${tempoBpm}</per-minute></metronome></direction-type><sound tempo="${tempoBpm}"/></direction><note><rest/><duration>${DIVISIONS * beatsPerMeasure}</duration></note></measure></part></score-partwise>`
        );
    }

    const getNoteType = (quarterLength: number): string => {
        if (quarterLength >= 4) return 'whole';
        if (quarterLength >= 2) return 'half';
        if (quarterLength >= 1) return 'quarter';
        if (quarterLength >= 0.5) return 'eighth';
        if (quarterLength >= 0.25) return '16th';
        if (quarterLength >= 0.125) return '32nd';
        return '64th';
    };

    const expandedNotes = sanitizedNotes.flatMap(note => {
        const noteSegments: Array<{ base: NoteEvent; segmentStart: number; segmentBeats: number; tieTypes: Array<'start' | 'stop'> }> = [];
        let remainingBeats = note.durationBeats;
        let segmentStart = note.startBeat;
        let first = true;

        while (remainingBeats > 1e-4) {
            const measureIdx = Math.floor(segmentStart / beatsPerMeasure);
            const measureEnd = (measureIdx + 1) * beatsPerMeasure;
            const span = Math.min(remainingBeats, measureEnd - segmentStart);
            const tieTypes: Array<'start' | 'stop'> = [];
            const continues = remainingBeats - span > 1e-4;
            if (continues) tieTypes.push('start');
            if (!first) tieTypes.push('stop');
            noteSegments.push({ base: note, segmentStart, segmentBeats: span, tieTypes });
            remainingBeats -= span;
            segmentStart += span;
            first = false;
        }

        return noteSegments;
    });

    const totalBeats = Math.max(...expandedNotes.map(n => n.segmentStart + n.segmentBeats));
    const totalMeasures = Math.max(1, Math.ceil(totalBeats / beatsPerMeasure));

    let currentNoteIdx = 0;
    let currentChordIdx = 0;

    for (let m = 1; m <= totalMeasures; m++) {
        const measureBeatStart = (m - 1) * beatsPerMeasure;
        const measureBeatEnd = m * beatsPerMeasure;
        let cursorBeat = measureBeatStart;

        xml += `<measure number="${m}">`;

        if (m === 1) {
            xml += `<attributes>
            <divisions>${DIVISIONS}</divisions>
            <key><fifths>0</fifths></key>
            <time><beats>${beatsPerMeasure}</beats><beat-type>${beatType}</beat-type></time>
            <clef><sign>G</sign><line>2</line></clef>
            </attributes>`;
            xml += `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${tempoBpm}</per-minute></metronome></direction-type><sound tempo="${tempoBpm}"/></direction>`;
        }

        const insertChordIfDue = (currentBeat: number) => {
            while (currentChordIdx < chordBeats.length) {
                const chord = chordBeats[currentChordIdx];
                if ((chord.startBeat ?? 0) < currentBeat - 0.0001) {
                    currentChordIdx++;
                    continue;
                }
                if ((chord.startBeat ?? 0) > currentBeat + 0.0001) break;
                xml += getMusicXMLHarmony(chord);
                currentChordIdx++;
                break;
            }
        };

        while (currentNoteIdx < expandedNotes.length) {
            const noteSegment = expandedNotes[currentNoteIdx];
            if (noteSegment.segmentStart >= measureBeatEnd) break;

            insertChordIfDue(cursorBeat);

            if (noteSegment.segmentStart > cursorBeat + 1e-4) {
                let restBeats = Math.min(noteSegment.segmentStart - cursorBeat, measureBeatEnd - cursorBeat);
                while (restBeats > 0.0001) {
                    const segment = Math.min(restBeats, beatsPerMeasure);
                    const restTicks = Math.max(1, Math.round(segment * DIVISIONS));
                    xml += `<note><rest/><duration>${restTicks}</duration><type>${getNoteType(segment)}</type></note>`;
                    cursorBeat += segment;
                    restBeats -= segment;
                    insertChordIfDue(cursorBeat);
                }
            }

            const stepNames = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B'];
            const alter = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0];
            const midi = Math.round(noteSegment.base.midi_pitch);
            const stepIndex = midi % 12;
            const octave = Math.floor(midi / 12) - 1;

            const clampedBeats = Math.max(noteSegment.segmentBeats, 1 / DIVISIONS);
            const durationTicks = Math.max(1, Math.round(clampedBeats * DIVISIONS));
            const type = getNoteType(clampedBeats);

            const tieTag = noteSegment.tieTypes.map(t => `<tie type="${t}"/>`).join('');
            const notationTie = noteSegment.tieTypes.length ? `<notations>${noteSegment.tieTypes.map(t => `<tie type="${t}"/>`).join('')}</notations>` : '';

            xml += `<note>
                <pitch>
                    <step>${stepNames[stepIndex]}</step>
                    <alter>${alter[stepIndex]}</alter>
                    <octave>${octave}</octave>
                </pitch>
                <duration>${durationTicks}</duration>
                <type>${type}</type>
                ${tieTag}
                ${notationTie}
            </note>`;

            cursorBeat = noteSegment.segmentStart + clampedBeats;
            currentNoteIdx++;
        }

        if (cursorBeat < measureBeatEnd - 1e-4) {
            const remainingBeats = measureBeatEnd - cursorBeat;
            const restTicks = Math.max(1, Math.round(remainingBeats * DIVISIONS));
            xml += `<note><rest/><duration>${restTicks}</duration><type>${getNoteType(remainingBeats)}</type></note>`;
        }

        xml += `</measure>`;
    }

    xml += `</part></score-partwise>`;
    return xml;
};

export interface TranscriptionError {
    message: string;
    stack?: string;
}

export interface TranscriptionResult {
    xml: string;
    musicxml?: string;
    midiBytes: Uint8Array;
    notes: NoteEvent[];
    quantizedNotes: NoteEvent[];
    preQuantizedNotes: NoteEvent[];
    chords: ChordEvent[];
    analysis: AnalysisDataLite;
    error: TranscriptionError | null;
}

export const TranscriptionService = {
  transcribeAudio: async (audioBlob: Blob): Promise<TranscriptionResult> => {
    // Client-Side Only flow for consistent Play Notes functionality
    // (Bypassing backend to ensure we have the NoteEvent objects needed for the sequencer)
    console.log("Using local transcription engine...");
    
    try {
        // Use a fresh context for decoding to avoid state issues
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        
        // Analyze audio to get notes using the existing engine logic
        const preQuantizedNotes = audioEngine.analyzeAudioSegment(audioBuffer, 0, audioBuffer.duration);

        // Quantize for cleaner notation
        const tempoBpm = 120;
        const timeSignature = '4/4';
        const beatsPerMeasure = Number(timeSignature.split('/')[0]) || 4;
        const secondsPerBeat = 60 / tempoBpm;

        const quantizedSeconds = audioEngine.cleanupAndQuantize(preQuantizedNotes);
        const quantizedNotes: NoteEvent[] = quantizedSeconds.map(n => {
            const startBeat = n.start_time / secondsPerBeat;
            const durationBeats = Math.max(n.duration / secondsPerBeat, 0.0625);
            const measure = Math.floor(startBeat / beatsPerMeasure) + 1;
            const beat = (startBeat % beatsPerMeasure) + 1;
            return {
                ...n,
                startBeat,
                durationBeats,
                measure,
                beat
            };
        });

        // Analyze Chords from quantized notes (aligned to beats)
        const chords = detectChords(quantizedSeconds).map(chord => {
            const startBeat = chord.time / secondsPerBeat;
            const measure = Math.floor(startBeat / beatsPerMeasure) + 1;
            const beat = (startBeat % beatsPerMeasure) + 1;
            return {
                ...chord,
                startBeat,
                measure,
                beat
            };
        });

        // Convert notes to MusicXML
        const xml = createMusicXML(quantizedNotes, chords, { tempoBpm, timeSignature });

        // Placeholder MIDI bytes (client-side fallback)
        const midiBytes = new Uint8Array();

        const meta: TranscriptionMeta = {
            tempo_bpm: tempoBpm,
            time_signature: timeSignature,
            tempo_override: null,
            beat_times_override: null
        };

        const analysis: AnalysisDataLite = {
            meta,
            events: quantizedNotes,
            chords,
            notes: quantizedNotes,
            timeline: [],
            vexflow_layout: { measures: [] }
        };

        return {
            xml,
            musicxml: xml,
            midiBytes,
            notes: quantizedNotes,
            quantizedNotes,
            preQuantizedNotes,
            chords,
            analysis,
            error: null
        };
    } catch (clientError) {
        console.error("Local transcription failed:", clientError);
        // Return valid empty score to prevent UI crash
        return {
            xml: `<?xml version="1.0" encoding="UTF-8"?><score-partwise><part-list><score-part id="P1"><part-name>Error</part-name></score-part></part-list><part id="P1"><measure number="1"><note><rest/><duration>4</duration></note></measure></part></score-partwise>` ,
            musicxml: undefined,
            midiBytes: new Uint8Array(),
            notes: [],
            quantizedNotes: [],
            preQuantizedNotes: [],
            chords: [],
            analysis: {
                meta: { tempo_bpm: 120, time_signature: '4/4', tempo_override: null, beat_times_override: null },
                events: [],
                chords: []
            },
            error: {
                message: clientError instanceof Error ? clientError.message : 'Unknown transcription error',
                stack: clientError instanceof Error ? clientError.stack : undefined
            }
        };
    }
  }
};