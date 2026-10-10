"""Central configuration for the AI Music Generator.

Every tunable value lives here so the rest of the code contains no magic
numbers. Change a default below (or build a modified copy with
``dataclasses.replace``) instead of editing the core logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Quieter TensorFlow logs and consistent numerics. These must be set before
# TensorFlow is imported anywhere, which is why they live in this module.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# --------------------------------------------------------------------------
# Paths (all relative to the project root, so they work on any machine)
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODEL_DIR: Path = PROJECT_ROOT / "models"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# --------------------------------------------------------------------------
# File names
# --------------------------------------------------------------------------
MIDI_EXTENSIONS: tuple[str, ...] = (".mid", ".midi")
PROCESSED_DATA_FILENAME: str = "encoded_tokens.npz"
VOCAB_FILENAME: str = "vocabulary.json"
PREPROCESS_REPORT_FILENAME: str = "preprocess_report.json"
BEST_MODEL_FILENAME: str = "music_lstm_best.keras"
FINAL_MODEL_FILENAME: str = "music_lstm_final.keras"
MODEL_CONFIG_FILENAME: str = "model_config.json"
HISTORY_FILENAME: str = "training_history.json"
SUMMARY_FILENAME: str = "training_summary.json"
HISTORY_PLOT_FILENAME: str = "training_history.png"
LOG_FILENAME: str = "music_generator.log"
OUTPUT_PREFIX: str = "generated_music"

# MIDI header that every valid .mid file starts with.
MIDI_HEADER: bytes = b"MThd"


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for turning MIDI files into token sequences."""

    min_token_count: int = 2  # tokens rarer than this are dropped (keeps vocab small)
    max_chord_size: int = 6  # chords with more notes keep only the lowest N notes
    min_tokens_per_file: int = 16  # files with fewer usable tokens are skipped
    max_file_mb: float = 10.0  # larger files are skipped (very slow to parse)


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for building and training the LSTM model."""

    sequence_length: int = 50
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    validation_split: float = 0.1
    random_seed: int = 42
    embedding_dim: int = 64
    lstm_units: int = 128
    dropout: float = 0.3
    min_sequences: int = 50  # fewer training windows than this -> clear error
    max_sequences: int = 200_000  # cap so student laptops do not run out of RAM
    early_stopping_patience: int = 4
    lr_reduce_patience: int = 2


@dataclass(frozen=True)
class GenerationConfig:
    """Settings for generating new music."""

    temperature: float = 1.0
    num_notes: int = 200
    random_seed: Optional[int] = None
    note_step: float = 0.5  # length of each generated event in quarter notes
    tempo_bpm: int = 120
    min_temperature: float = 0.1
    max_temperature: float = 2.0
    min_notes: int = 8
    max_notes: int = 2000


DEFAULT_PREPROCESS = PreprocessConfig()
DEFAULT_TRAINING = TrainingConfig()
DEFAULT_GENERATION = GenerationConfig()
