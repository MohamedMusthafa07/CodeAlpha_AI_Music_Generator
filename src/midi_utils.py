"""MIDI helpers: file discovery, token <-> music21 conversion and MIDI export.

Token format used everywhere in the project:
    single note -> "C4", "F#3"
    chord       -> "[C4.E4.G4]"   (notes sorted from low to high)
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from music21 import instrument, note, chord, pitch, stream, tempo

from src import config
from src.utils import GenerationError, PathLike, ensure_dir

logger = logging.getLogger(__name__)


def find_midi_files(directory: PathLike) -> list[Path]:
    """Recursively find .mid/.midi files (case-insensitive), sorted for determinism."""
    root = Path(directory)
    if not root.is_dir():
        logger.warning("MIDI folder does not exist: %s", root)
        return []
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in config.MIDI_EXTENSIONS
    ]
    return sorted(files, key=lambda p: str(p).lower())


def file_sha256(path: PathLike) -> str:
    """Return the SHA-256 hash of a file (used to detect duplicate files)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def has_midi_header(path: PathLike) -> bool:
    """Quick check that a file starts with the standard MIDI header bytes."""
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == config.MIDI_HEADER
    except OSError:
        return False

@lru_cache(maxsize=256)
def midi_to_name(midi_number: int) -> str:
    """Convert a MIDI pitch number to a name such as 'C#4'."""
    return pitch.Pitch(midi_number).nameWithOctave


def pitches_to_token(midi_numbers: Iterable[int]) -> str:
    """Build a token from one or more MIDI pitch numbers."""
    unique = sorted({int(m) for m in midi_numbers})
    if not unique:
        raise ValueError("Cannot build a token from zero pitches.")
    names = [midi_to_name(m) for m in unique]
    if len(names) == 1:
        return names[0]
    return "[" + ".".join(names) + "]"


def is_chord_token(token: str) -> bool:
    """Return True if the token represents a chord."""
    return token.startswith("[") and token.endswith("]")


def token_to_midis(token: str) -> list[int]:
    """Convert a token back to MIDI pitch numbers. Raises ValueError if invalid."""
    text = token.strip()
    parts = text[1:-1].split(".") if is_chord_token(text) else [text]
    numbers: list[int] = []
    for part in parts:
        try:
            number = int(pitch.Pitch(part).midi)
        except Exception as exc:  # music21 raises several unrelated exception types
            raise ValueError(f"Invalid pitch '{part}' in token '{token}'.") from exc
        if not 0 <= number <= 127:
            raise ValueError(f"Pitch out of MIDI range in token '{token}'.")
        numbers.append(number)
    if not numbers:
        raise ValueError(f"Token '{token}' contains no pitches.")
    return sorted(set(numbers))


def token_to_element(token: str) -> note.Note | chord.Chord:
    """Convert a token into a music21 Note or Chord."""
    midis = token_to_midis(token)
    if len(midis) == 1:
        return note.Note(midis[0])
    return chord.Chord(midis)


def tokens_to_stream(
    tokens: Sequence[str],
    note_step: float = config.DEFAULT_GENERATION.note_step,
    tempo_bpm: int = config.DEFAULT_GENERATION.tempo_bpm,
) -> tuple[stream.Score, int]:
    """Build a music21 Score from tokens; returns (score, number_of_events_used)."""
    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.insert(0, tempo.MetronomeMark(number=tempo_bpm))

    offset = 0.0
    used = 0
    for token in tokens:
        try:
            element = token_to_element(token)
        except ValueError as exc:
            logger.warning("Skipping invalid token during MIDI creation: %s", exc)
            continue
        element.quarterLength = note_step
        part.insert(offset, element)
        offset += note_step
        used += 1

    if used == 0:
        raise GenerationError(
            "No playable notes were produced.",
            reason="Every generated token was invalid.",
            fix="Retrain the model with more data and try a lower temperature.",
        )
    score = stream.Score()
    score.insert(0, part)
    return score, used


def estimate_duration_seconds(num_events: int, note_step: float, tempo_bpm: int) -> float:
    """Approximate playing time in seconds for evenly spaced events."""
    return num_events * note_step * 60.0 / tempo_bpm



def _unique_path(path: Path) -> Path:
    """Return a path that does not exist yet by adding _1, _2, ... if needed."""
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def write_midi_file(
    score: stream.Stream,
    output_dir: PathLike = config.OUTPUT_DIR,
    filename: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a score to a MIDI file and verify the file really exists."""
    directory = ensure_dir(output_dir)
    if filename is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.OUTPUT_PREFIX}_{stamp}.mid"
    if not filename.lower().endswith(config.MIDI_EXTENSIONS):
        filename += ".mid"
    
    target = directory / Path(filename).name
    if not overwrite:
        target = _unique_path(target)

    try:
        score.write("midi", fp=str(target))
    except Exception as exc:  # music21 can raise many different exception types
        logger.exception("MIDI export failed")
        raise GenerationError(
            "The MIDI file could not be created.",
            reason=f"music21 reported: {exc}",
            fix="Check that the outputs/ folder is writable and the file is not open in a player.",
        ) from exc

    if not target.exists() or target.stat().st_size == 0:
        raise GenerationError(
            "The MIDI file was not created correctly.",
            reason="The file is missing or empty after export.",
            fix="Check disk space and folder permissions for the outputs/ folder, then try again.",
        )
    return target


def find_soundfont() -> Path | None:
    """Return a SoundFont (.sf2) from MUSIC_SOUNDFONT or models/soundfont.sf2, if present."""
    candidates = [os.environ.get("MUSIC_SOUNDFONT", ""), str(config.MODEL_DIR / "soundfont.sf2")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def playback_available() -> bool:
    """True only if FluidSynth and a SoundFont are both available."""
    return shutil.which("fluidsynth") is not None and find_soundfont() is not None


def midi_to_wav(midi_path: PathLike, wav_path: PathLike | None = None) -> Path:
    """Convert MIDI to WAV with FluidSynth (optional feature; never required)."""
    fluidsynth = shutil.which("fluidsynth")
    soundfont = find_soundfont()
    if fluidsynth is None or soundfont is None:
        raise GenerationError(
            "Audio playback is not available.",
            reason="FluidSynth or a SoundFont (.sf2) file was not found.",
            fix="Open the downloaded MIDI file with any MIDI-compatible player instead.",
        )
    source = Path(midi_path)
    target = Path(wav_path) if wav_path else source.with_suffix(".wav")
    # Arguments are passed as a list (no shell), and paths come from the app itself.
    command = [fluidsynth, "-ni", str(soundfont), str(source), "-F", str(target), "-r", "44100"]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as exc:
        raise GenerationError(
            "Audio conversion failed.",
            reason=str(exc),
            fix="Open the MIDI file with a MIDI-compatible player instead.",
        ) from exc
    return target
