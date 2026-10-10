import numpy as np
import pytest

from src import config
from src.model import build_model, validate_shapes
from src.train import train_model
from src.utils import DatasetError, TrainingError


def test_model_output_shape_and_compilation():
    model = build_model(vocab_size=30, sequence_length=10, embedding_dim=8, lstm_units=16)
    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 30)
    assert model.optimizer is not None and model.loss is not None
    probs = model.predict(np.zeros((2, 10), dtype=np.int32), verbose=0)
    assert probs.shape == (2, 30)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)  # softmax output


def test_tiny_vocabulary_is_rejected():
    with pytest.raises(TrainingError):
        build_model(vocab_size=1, sequence_length=10)


def test_validate_shapes_reports_mismatch_clearly():
    model = build_model(vocab_size=20, sequence_length=10, embedding_dim=8, lstm_units=8)
    good_x = np.zeros((4, 10), dtype=np.int32)
    good_y = np.zeros(4, dtype=np.int32)
    validate_shapes(model, good_x, good_y, 20)  # no error
    with pytest.raises(TrainingError):
        validate_shapes(model, np.zeros((4, 7), dtype=np.int32), good_y, 20)
    with pytest.raises(TrainingError):
        validate_shapes(model, good_x, np.full(4, 99, dtype=np.int32), 20)


def test_training_smoke_test_saves_files(processed_dir, tmp_path):
    """One quick epoch on tiny data proves the whole training path works."""
    from dataclasses import replace

    cfg = replace(
        config.DEFAULT_TRAINING, sequence_length=10, epochs=1, batch_size=32,
        lstm_units=16, embedding_dim=8, min_sequences=10,
    )
    models = tmp_path / "models"
    summary = train_model(cfg, processed_dir, models)
    assert (models / config.BEST_MODEL_FILENAME).exists()
    assert (models / config.MODEL_CONFIG_FILENAME).exists()
    assert summary["epochs_run"] == 1 and summary["best_val_loss"] > 0


def test_training_without_processed_data_gives_helpful_error(tmp_path):
    with pytest.raises(Exception) as info:
        train_model(config.DEFAULT_TRAINING, tmp_path / "none", tmp_path / "models")
    assert "preprocessing" in str(getattr(info.value, "fix", "")).lower() or isinstance(info.value, DatasetError)
