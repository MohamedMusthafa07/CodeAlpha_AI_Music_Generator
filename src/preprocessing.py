"""MIDI dataset scanning and preprocessing with music21.

Run from the project root:
    python -m src.preprocessing
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from music21 import chord, converter, note

from src import config
from src.dataset import build_vocabulary, encode_tokens, save_processed, save_vocabulary
from src.midi_utils import (
    file_sha256,
    find_midi_files,
    has_midi_header,
    pitches_to_token,
)
from src.utils import (
    DatasetError,
    MidiParseError,
    MusicGenError,
    PathLike,
    ensure_dir,
    save_json,
    setup_logging,
)

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass
class ScanReport:
    """Result of a quick (non-parsing) scan of the dataset folder."""

    directory: str
    files_found: int = 0
    valid_files: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return len(self.valid_files)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["valid_count"] = self.valid_count
        data["skipped_count"] = self.skipped_count
        return data


def scan_dataset(
    raw_dir: PathLike = config.RAW_DATA_DIR,
    cfg: config.PreprocessConfig = config.DEFAULT_PREPROCESS,
) -> ScanReport:
    """Find MIDI files and run cheap checks (empty, header, size, duplicates)."""
    root = Path(raw_dir)
    report = ScanReport(directory=str(root))
    seen_hashes: dict[str, str] = {}
    max_bytes = cfg.max_file_mb * 1024 * 1024

    for path in find_midi_files(root):
        report.files_found += 1
        label = _relative_label(path, root)
        try:
            size = path.stat().st_size
        except OSError as exc:
            report.skipped.append({"file": label, "reason": f"cannot read file ({exc})"})
            continue
        if size == 0:
            report.skipped.append({"file": label, "reason": "empty file"})
        elif size > max_bytes:
            report.skipped.append({"file": label, "reason": f"larger than {cfg.max_file_mb} MB"})
        elif not has_midi_header(path):
            report.skipped.append({"file": label, "reason": "not a valid MIDI file (bad header)"})
        else:
            try:
                digest = file_sha256(path)
            except OSError as exc:
                report.skipped.append({"file": label, "reason": f"cannot read file ({exc})"})
                continue
            if digest in seen_hashes:
                report.skipped.append(
                    {"file": label, "reason": f"duplicate of {seen_hashes[digest]}"}
                )
            else:
                seen_hashes[digest] = label
                report.valid_files.append(label)

    for item in report.skipped:
        logger.warning("Skipped %s: %s", item["file"], item["reason"])
    logger.info(
        "Scan complete: %d found, %d usable, %d skipped",
        report.files_found, report.valid_count, report.skipped_count,
    )
    return report


def _relative_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def extract_tokens(path: PathLike, max_chord_size: int = config.DEFAULT_PREPROCESS.max_chord_size) -> list[str]:
    """Parse one MIDI file with music21 and return its note/chord tokens.

    Notes that start at the same moment are merged into one chord token, which
    gives a clean one-token-per-time-step sequence for the LSTM.
    """
    try:
        score = converter.parse(str(path), forceSource=True)
        elements = score.flatten().notes
    except Exception as exc:  # music21 raises many unrelated exception types
        raise MidiParseError(
            f"Could not parse '{Path(path).name}'.",
            reason=f"music21 reported: {exc}",
            fix="The file may be corrupted or use an unsupported MIDI structure; it will be skipped.",
        ) from exc

    by_offset: dict[float, set[int]] = defaultdict(set)
    for element in elements:
        offset = round(float(element.offset), 4)
        if isinstance(element, note.Note):
            by_offset[offset].add(int(element.pitch.midi))
        elif isinstance(element, chord.Chord):
            for chord_pitch in element.pitches:
                by_offset[offset].add(int(chord_pitch.midi))

    tokens: list[str] = []
    for offset in sorted(by_offset):
        midis = sorted(m for m in by_offset[offset] if 0 <= m <= 127)
        if midis:
            tokens.append(pitches_to_token(midis[:max_chord_size]))
    return tokens


def preprocess_dataset(
    raw_dir: PathLike = config.RAW_DATA_DIR,
    processed_dir: PathLike = config.PROCESSED_DIR,
    cfg: config.PreprocessConfig = config.DEFAULT_PREPROCESS,
    progress: ProgressCallback = None,
) -> dict:
    """Run the full preprocessing pipeline and save the processed data.

    Returns a report dictionary (also saved to preprocess_report.json).
    """
    root = Path(raw_dir)
    scan = scan_dataset(root, cfg)
    if scan.files_found == 0:
        raise DatasetError(
            "No MIDI files were found.",
            reason=f"The folder '{root}' is empty or does not exist.",
            fix="Add legally obtained .mid or .midi files to data/raw/ "
            "(or run 'python -m src.sample_data' to create demo files).",
        )
    if scan.valid_count == 0:
        raise DatasetError(
            "None of the MIDI files could be used.",
            reason="All files were empty, damaged, too large or duplicates.",
            fix="Check the skipped-file list in logs/ and replace them with valid MIDI files.",
        )

    token_lists: list[list[str]] = []
    skipped = list(scan.skipped)
    total = scan.valid_count
    for index, label in enumerate(scan.valid_files, start=1):
        if progress:
            progress(index, total, label)
        try:
            tokens = extract_tokens(root / label, cfg.max_chord_size)
        except MidiParseError as exc:
            logger.warning("Skipped %s: %s", label, exc.reason)
            skipped.append({"file": label, "reason": "could not be parsed"})
            continue
        if len(tokens) < cfg.min_tokens_per_file:
            logger.warning("Skipped %s: only %d usable notes", label, len(tokens))
            skipped.append({"file": label, "reason": f"too few usable notes ({len(tokens)})"})
            continue
        token_lists.append(tokens)

    if not token_lists:
        raise DatasetError(
            "No usable notes could be extracted.",
            reason="Every file was unreadable or contained too few notes.",
            fix="Use MIDI files containing melodies or chords (not only drums).",
        )

    counts = Counter(token for tokens in token_lists for token in tokens)
    vocabulary = build_vocabulary(counts, cfg.min_token_count)
    token_to_id = {token: index for index, token in enumerate(vocabulary)}

    encoded = [encode_tokens(tokens, token_to_id) for tokens in token_lists]
    encoded = [arr for arr in encoded if len(arr) > 0]
    if not encoded:
        raise DatasetError(
            "No tokens remained after filtering rare notes.",
            reason="The dataset is too small for the minimum token count.",
            fix="Add more MIDI files.",
        )
    lengths = np.array([len(arr) for arr in encoded], dtype=np.int64)
    ids = np.concatenate(encoded).astype(np.int32)

    out_dir = ensure_dir(processed_dir)
    save_processed(out_dir / config.PROCESSED_DATA_FILENAME, ids, lengths)
    save_vocabulary(out_dir / config.VOCAB_FILENAME, vocabulary, cfg.min_token_count)

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "files_found": scan.files_found,
        "files_used": len(encoded),
        "files_skipped": len(skipped) + (len(token_lists) - len(encoded)),
        "skipped": skipped,
        "total_tokens": int(ids.size),
        "vocabulary_size": len(vocabulary),
    }
    save_json(out_dir / config.PREPROCESS_REPORT_FILENAME, report)
    logger.info(
        "Preprocessing finished: %d files used, %d tokens, vocabulary %d",
        report["files_used"], report["total_tokens"], report["vocabulary_size"],
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess MIDI files for training.")
    parser.add_argument("--raw-dir", default=str(config.RAW_DATA_DIR), help="Folder with .mid/.midi files")
    parser.add_argument("--processed-dir", default=str(config.PROCESSED_DIR), help="Where to save processed data")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        report = preprocess_dataset(
            args.raw_dir,
            args.processed_dir,
            progress=lambda i, n, name: print(f"[{i}/{n}] {name}"),
        )
    except MusicGenError as exc:
        logger.error(exc.message)
        print("\n" + exc.describe())
        return 1
    print(
        f"\nDone. Files used: {report['files_used']} | Skipped: {report['files_skipped']} | "
        f"Tokens: {report['total_tokens']} | Vocabulary: {report['vocabulary_size']}"
    )
    print("Next step: python -m src.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
