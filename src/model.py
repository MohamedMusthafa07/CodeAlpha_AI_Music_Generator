"""LSTM model definition (TensorFlow / Keras)."""

from __future__ import annotations

import logging

from src import config  # noqa: F401  (sets TensorFlow log levels before import)
from src.utils import TrainingError

try:
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise TrainingError(
        "TensorFlow is not installed.",
        reason=str(exc),
        fix="Activate your virtual environment and run: pip install -r requirements.txt",
    ) from exc

logger = logging.getLogger(__name__)


def build_model(
    vocab_size: int,
    sequence_length: int = config.DEFAULT_TRAINING.sequence_length,
    embedding_dim: int = config.DEFAULT_TRAINING.embedding_dim,
    lstm_units: int = config.DEFAULT_TRAINING.lstm_units,
    dropout: float = config.DEFAULT_TRAINING.dropout,
    learning_rate: float = config.DEFAULT_TRAINING.learning_rate,
) -> keras.Model:
    """Build and compile: Embedding -> LSTM -> Dropout -> LSTM -> Dropout -> Dense(softmax)."""
    if vocab_size < 2:
        raise TrainingError(
            "The vocabulary is too small to train a model.",
            reason=f"Only {vocab_size} unique token(s) were found.",
            fix="Add more varied MIDI files and run preprocessing again.",
        )
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length,), dtype="int32"),
            keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            keras.layers.LSTM(lstm_units, return_sequences=True),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(lstm_units),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(vocab_size, activation="softmax"),
        ],
        name="music_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def validate_shapes(model: keras.Model, x, y, vocab_size: int) -> None:
    """Check data and model agree, with a clear message instead of a cryptic crash."""
    expected_len = model.input_shape[1]
    if x.ndim != 2 or x.shape[1] != expected_len:
        raise TrainingError(
            "Input data has the wrong shape.",
            reason=f"Model expects sequences of length {expected_len}, data has shape {tuple(x.shape)}.",
            fix="Re-create the sequences with the same sequence length used to build the model.",
        )
    if len(x) != len(y):
        raise TrainingError(
            "Inputs and targets have different lengths.",
            reason=f"{len(x)} inputs vs {len(y)} targets.",
            fix="Re-run training; if it persists, re-run preprocessing.",
        )
    if int(x.max()) >= vocab_size or int(y.max()) >= vocab_size or int(x.min()) < 0:
        raise TrainingError(
            "Token IDs are outside the vocabulary range.",
            reason="The processed data and vocabulary do not match.",
            fix="Run preprocessing again: python -m src.preprocessing",
        )
    if model.output_shape[-1] != vocab_size:
        raise TrainingError(
            "Model output size does not match the vocabulary.",
            reason=f"Model outputs {model.output_shape[-1]} classes, vocabulary has {vocab_size}.",
            fix="Rebuild the model with the current vocabulary size.",
        )
