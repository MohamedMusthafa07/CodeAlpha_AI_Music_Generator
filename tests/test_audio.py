import io
import wave

import numpy as np
import pytest

from src.audio_utils import SAMPLE_RATE, midi_to_wav_bytes, read_midi_notes, synthesize
from src.utils import GenerationError


def test_midi_converts_to_valid_wav(raw_dir):
    midi = next(raw_dir.glob("*.mid"))
    data = midi_to_wav_bytes(midi)
    with wave.open(io.BytesIO(data)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getsampwidth() == 2
        assert handle.getnframes() / SAMPLE_RATE > 20  # 120 events x 0.25 s each
    samples = np.frombuffer(data[44:], dtype="<i2")
    assert np.abs(samples).max() > 1000  # not silent


def test_a4_note_has_440_hz_peak():
    audio = synthesize([(0.0, 1.0, 69)])
    spectrum = np.abs(np.fft.rfft(audio))
    peak_hz = np.argmax(spectrum) * SAMPLE_RATE / len(audio)
    assert abs(peak_hz - 440) < 5


def test_read_notes_matches_file(raw_dir):
    notes = read_midi_notes(next(raw_dir.glob("*.mid")))
    assert len(notes) >= 100
    assert all(0 <= midi <= 127 and length > 0 for _, length, midi in notes)


def test_missing_file_gives_friendly_error(tmp_path):
    with pytest.raises(GenerationError) as info:
        midi_to_wav_bytes(tmp_path / "nope.mid")
    assert info.value.fix


def test_empty_note_list_rejected():
    with pytest.raises(GenerationError):
        synthesize([])


def test_overlong_composition_rejected():
    with pytest.raises(GenerationError):
        synthesize([(0.0, 5000.0, 60)])