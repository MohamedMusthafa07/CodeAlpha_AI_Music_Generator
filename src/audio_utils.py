"""Turn a MIDI file into a playable WAV preview with a small built-in synthesizer.

Browsers cannot play MIDI directly, so the app renders the notes to audio itself.
No extra software (FluidSynth, SoundFonts) is needed. The sound is a simple
piano-like tone, so for richer instruments open the .mid file in a MIDI player.
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path

import numpy as np
from music21 import chord, converter, note

from src.utils import GenerationError, PathLike

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050
HARMONICS = ((1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12))  # (multiple of the base frequency, loudness)
ATTACK_SECONDS = 0.01
DECAY_SECONDS = 0.45
RELEASE_SECONDS = 0.25
MAX_SECONDS = 600.0
DEFAULT_BPM = 120.0

NoteList = list[tuple[float, float, int]]  # (start seconds, length seconds, MIDI pitch)


def read_midi_notes(path: PathLike) -> NoteList:
    """Read a MIDI file with music21 and return its notes in seconds."""
    source = Path(path)
    if not source.is_file():
        raise GenerationError(
            "The MIDI file was not found.",
            reason=f"'{source.name}' does not exist in outputs/.",
            fix="Generate the music again.",
        )
    try:
        flat = converter.parse(str(source), forceSource=True).flatten()
        marks = list(flat.getElementsByClass("MetronomeMark"))
        bpm = float(marks[0].number) if marks and marks[0].number else DEFAULT_BPM
        seconds_per_quarter = 60.0 / bpm
        notes: NoteList = []
        for element in flat.notes:
            start = float(element.offset) * seconds_per_quarter
            length = max(float(element.quarterLength), 0.05) * seconds_per_quarter
            if isinstance(element, note.Note):
                notes.append((start, length, int(element.pitch.midi)))
            elif isinstance(element, chord.Chord):
                notes.extend((start, length, int(p.midi)) for p in element.pitches)
    except Exception as exc:  # music21 raises many unrelated exception types
        logger.exception("Could not read MIDI for audio preview")
        raise GenerationError(
            "The audio preview could not be created.",
            reason=f"The MIDI file could not be read: {exc}",
            fix="You can still download the MIDI file and open it in a MIDI player.",
        ) from exc
    return notes


def synthesize(notes: NoteList, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Render notes to mono float audio in the range -1..1."""
    if not notes:
        raise GenerationError(
            "There are no notes to play.",
            reason="The MIDI file contains no notes.",
            fix="Generate the music again.",
        )
    total = max(start + length for start, length, _ in notes) + RELEASE_SECONDS
    if total > MAX_SECONDS:
        raise GenerationError(
            "The composition is too long for a preview.",
            reason=f"It would last {total:.0f} seconds (limit {MAX_SECONDS:.0f}).",
            fix="Generate fewer notes, or open the MIDI file in a MIDI player.",
        )

    audio = np.zeros(int(total * sample_rate) + 1, dtype=np.float32)
    for start, length, midi in notes:
        frequency = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        duration = length + RELEASE_SECONDS
        t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
        tone = np.zeros_like(t)
        for multiple, loudness in HARMONICS:
            if frequency * multiple < sample_rate / 2:
                tone += loudness * np.sin(2.0 * np.pi * frequency * multiple * t)
        envelope = np.minimum(t / ATTACK_SECONDS, 1.0) * np.exp(-t / DECAY_SECONDS)
        release = np.clip((duration - t) / RELEASE_SECONDS, 0.0, 1.0)
        begin = int(start * sample_rate)
        piece = (tone * envelope * release)[: len(audio) - begin]
        audio[begin : begin + len(piece)] += piece

    peak = float(np.abs(audio).max())
    if peak > 0:
        audio *= 0.85 / peak
    return audio


def to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode float audio as a 16-bit mono WAV file in memory."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def midi_to_wav_bytes(path: PathLike) -> bytes:
    """Full pipeline: MIDI file -> notes -> audio -> WAV bytes."""
    return to_wav_bytes(synthesize(read_midi_notes(path)))