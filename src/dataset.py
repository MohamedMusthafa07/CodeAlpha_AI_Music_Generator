"""Vocabulary, encoding and training-sequence creation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from src import config
from src.utils import DatasetError, PathLike, ValidationError, ensure_dir, load_json, save_json

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
def build_vocabulary(token_counts: Mapping[str, int], min_count: int = 1) -> list[str]:
    """Return a sorted (deterministic) token list, dropping rare tokens."""
    return sorted(token for token, count in token_counts.items() if count >= min_count)


def encode_tokens(tokens: Iterable[str], token_to_id: Mapping[str, int]) -> np.ndarray:
    """Convert tokens to integer IDs, silently dropping tokens not in the vocabulary."""
    return np.array([token_to_id[t] for t in tokens if t in token_to_id], dtype=np.int32)


def save_vocabulary(path: PathLike, vocabulary: Sequence[str], min_count: int = 1) -> None:
    """Save the vocabulary; the list index of each token IS its integer ID."""
    save_json(path, {"tokens": list(vocabulary), "min_token_count": min_count})


def load_vocabulary(path: PathLike) -> list[str]:
    """Load the vocabulary saved during preprocessing."""
    data = load_json(path)
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if not tokens:
        raise DatasetError(
            "The vocabulary file is empty or has the wrong format.",
            reason="It was probably damaged or created by a different version.",
            fix="Run preprocessing again: python -m src.preprocessing",
        )
    return [str(token) for token in tokens]


# --------------------------------------------------------------------------
# Processed data storage
# --------------------------------------------------------------------------
def save_processed(path: PathLike, ids: np.ndarray, lengths: np.ndarray) -> None:
    """Save all token IDs plus the length of each file's sequence."""
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, ids=ids.astype(np.int32), lengths=lengths.astype(np.int64))


def load_processed(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Load (ids, lengths) saved by preprocessing."""
    source = Path(path)
    if not source.exists():
        raise DatasetError(
            "Processed data was not found.",
            reason="Preprocessing has not been run yet.",
            fix="Run: python -m src.preprocessing",
        )
    try:
        with np.load(source) as data:
            ids = data["ids"]
            lengths = data["lengths"]
    except (OSError, ValueError, KeyError) as exc:
        raise DatasetError(
            "Processed data could not be read.",
            reason=f"The file is damaged: {exc}",
            fix="Run preprocessing again: python -m src.preprocessing",
        ) from exc
    if int(lengths.sum()) != int(ids.size):
        raise DatasetError(
            "Processed data is inconsistent.",
            reason="The stored lengths do not match the number of tokens.",
            fix="Run preprocessing again: python -m src.preprocessing",
        )
    return ids, lengths


# --------------------------------------------------------------------------
# Sequence creation
# --------------------------------------------------------------------------
def validate_sequence_length(sequence_length: int) -> int:
    """Make sure the sequence length is a sensible positive integer."""
    if not isinstance(sequence_length, (int, np.integer)) or isinstance(sequence_length, bool):
        raise ValidationError("Sequence length must be a whole number.", fix="Use a value such as 50.")
    if not 2 <= sequence_length <= 1000:
        raise ValidationError(
            f"Sequence length {sequence_length} is not allowed.",
            reason="It must be between 2 and 1000.",
            fix="Use a value such as 50.",
        )
    return int(sequence_length)


def _window_counts(lengths: np.ndarray, sequence_length: int) -> np.ndarray:
    """Number of (input, target) windows each file can provide."""
    return np.maximum(lengths - sequence_length, 0)


def create_sequences(
    ids: np.ndarray,
    lengths: np.ndarray,
    sequence_length: int,
    max_sequences: int | None = None,
    seed: int = config.DEFAULT_TRAINING.random_seed,
    min_sequences: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window training data without crossing file boundaries.

    Returns:
        X with shape (num_sequences, sequence_length) - integer token IDs
        y with shape (num_sequences,)                  - the next token ID
    """
    sequence_length = validate_sequence_length(sequence_length)
    counts = _window_counts(np.asarray(lengths, dtype=np.int64), sequence_length)
    total = int(counts.sum())
    if total < max(min_sequences, 1):
        raise DatasetError(
            f"Only {total} training sequences can be created (at least {max(min_sequences, 1)} needed).",
            reason=f"The dataset has too few notes for a sequence length of {sequence_length}.",
            fix="Add more MIDI files to data/raw/, re-run preprocessing, or lower the sequence length.",
        )

    file_starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    cumulative = np.cumsum(counts)

    if max_sequences is not None and total > max_sequences:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(total, size=max_sequences, replace=False))
        logger.info("Using %d of %d possible sequences", max_sequences, total)
    else:
        chosen = np.arange(total)

    file_index = np.searchsorted(cumulative, chosen, side="right")
    before = cumulative[file_index] - counts[file_index]
    starts = file_starts[file_index] + (chosen - before)

    positions = starts[:, None] + np.arange(sequence_length + 1)[None, :]
    windows = ids[positions]
    x = np.ascontiguousarray(windows[:, :-1], dtype=np.int32)
    y = np.ascontiguousarray(windows[:, -1], dtype=np.int32)
    return x, y


def train_val_split(
    x: np.ndarray, y: np.ndarray, validation_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle reproducibly and split into training and validation sets."""
    if not 0.0 < validation_split < 0.5:
        raise ValidationError(
            f"Validation split {validation_split} is not allowed.",
            reason="It must be greater than 0 and less than 0.5.",
            fix="Use a value such as 0.1.",
        )
    count = len(x)
    n_val = max(1, int(round(count * validation_split)))
    if count - n_val < 1:
        raise DatasetError(
            "Not enough sequences to create a validation set.",
            fix="Add more MIDI files to data/raw/.",
        )
    order = np.random.default_rng(seed).permutation(count)
    val_idx, train_idx = order[:n_val], order[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def pick_seed_window(
    ids: np.ndarray, lengths: np.ndarray, sequence_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Choose a random real window of the dataset to start generation from."""
    counts = _window_counts(np.asarray(lengths, dtype=np.int64), sequence_length - 1)
    total = int(counts.sum())
    if total < 1:
        raise DatasetError(
            "No seed sequence is available.",
            reason=f"No file has at least {sequence_length} tokens.",
            fix="Add longer MIDI files and re-run preprocessing and training.",
        )
    pick = int(rng.integers(total))
    cumulative = np.cumsum(counts)
    file_index = int(np.searchsorted(cumulative, pick, side="right"))
    file_start = int(np.concatenate(([0], np.cumsum(lengths)))[file_index])
    local = pick - int(cumulative[file_index] - counts[file_index])
    begin = file_start + local
    return ids[begin : begin + sequence_length].astype(np.int32)
