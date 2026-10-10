import json
import re
import shutil

import pytest

from src import config
from src.midi_utils import find_midi_files, pitches_to_token, token_to_midis
from src.preprocessing import extract_tokens, preprocess_dataset, scan_dataset
from src.utils import DatasetError, MidiParseError
from tests.conftest import SMALL_PREPROCESS

TOKEN_PATTERN = re.compile(r"^(\[?[A-G][#-]?\d(\.[A-G][#-]?\d)*\]?)$")


def test_find_midi_files_is_recursive_and_case_insensitive(raw_dir):
    nested = raw_dir / "sub" / "deeper"
    nested.mkdir(parents=True)
    first = next(raw_dir.glob("*.mid"))
    shutil.copy(first, nested / "UPPER.MID")
    shutil.copy(first, nested / "other.midi")
    (raw_dir / "notes.txt").write_text("not midi")
    names = {p.name for p in find_midi_files(raw_dir)}
    assert "UPPER.MID" in names and "other.midi" in names and "notes.txt" not in names


def test_find_midi_files_missing_folder_returns_empty(tmp_path):
    assert find_midi_files(tmp_path / "does_not_exist") == []


def test_scan_skips_empty_bad_header_and_duplicates(raw_dir):
    (raw_dir / "empty.mid").write_bytes(b"")
    (raw_dir / "fake.mid").write_bytes(b"this is not midi")
    shutil.copy(next(raw_dir.glob("demo_001.mid")), raw_dir / "copy_of_001.mid")
    report = scan_dataset(raw_dir)
    reasons = " | ".join(item["reason"] for item in report.skipped)
    assert report.valid_count == 4
    assert "empty file" in reasons and "bad header" in reasons and "duplicate" in reasons


def test_extract_tokens_returns_valid_tokens(raw_dir):
    tokens = extract_tokens(next(raw_dir.glob("*.mid")))
    assert len(tokens) > 50
    assert all(TOKEN_PATTERN.match(t) for t in tokens)


def test_corrupt_midi_raises_parse_error_and_is_skipped(raw_dir, tmp_path):
    corrupt = raw_dir / "corrupt.mid"
    corrupt.write_bytes(b"MThd" + b"\xff" * 40)
    with pytest.raises(MidiParseError):
        extract_tokens(corrupt)
    report = preprocess_dataset(raw_dir, tmp_path / "out", SMALL_PREPROCESS)
    assert report["files_used"] == 4
    assert any(item["file"] == "corrupt.mid" for item in report["skipped"])


def test_empty_dataset_gives_clear_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetError) as info:
        preprocess_dataset(empty, tmp_path / "out", SMALL_PREPROCESS)
    assert "data/raw" in info.value.fix


def test_preprocess_saves_all_artifacts(processed_dir):
    for name in (config.PROCESSED_DATA_FILENAME, config.VOCAB_FILENAME, config.PREPROCESS_REPORT_FILENAME):
        assert (processed_dir / name).exists()
    vocab = json.loads((processed_dir / config.VOCAB_FILENAME).read_text())["tokens"]
    assert vocab == sorted(vocab)  # deterministic ordering


def test_token_round_trip():
    token = pitches_to_token([64, 60, 67])
    assert token == "[C4.E4.G4]"
    assert token_to_midis(token) == [60, 64, 67]
    assert pitches_to_token([61]) == "C#4"
