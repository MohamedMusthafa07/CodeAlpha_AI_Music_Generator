import json

import numpy as np
import pytest

from src import config
from src.generate import (
    generate_music,
    generate_token_ids,
    load_generation_assets,
    sample_next_token,
    validate_num_notes,
    validate_temperature,
)
from src.model import build_model
from src.utils import ModelNotFoundError, ValidationError


class StubModel:
    """Tiny stand-in for a Keras model: always favours token 2."""

    def __call__(self, window, training=False):
        probs = np.full((1, 5), 0.05)
        probs[0, 2] = 0.8
        return probs


@pytest.mark.parametrize("bad", [0, -1, 5.0, float("nan"), "abc", None])
def test_invalid_temperature_rejected(bad):
    with pytest.raises(ValidationError):
        validate_temperature(bad)


def test_valid_temperature_accepted():
    assert validate_temperature(0.5) == 0.5 and validate_temperature("1.5") == 1.5


def test_invalid_note_count_rejected():
    with pytest.raises(ValidationError):
        validate_num_notes(1)
    assert validate_num_notes(100) == 100


def test_sampling_is_probabilistic_but_seeded():
    probs = np.array([0.1, 0.6, 0.3])
    draws = {sample_next_token(probs, 1.0, np.random.default_rng(s)) for s in range(50)}
    assert len(draws) > 1  # not always argmax
    a = sample_next_token(probs, 1.0, np.random.default_rng(9))
    b = sample_next_token(probs, 1.0, np.random.default_rng(9))
    assert a == b


def test_low_temperature_is_more_conservative():
    probs = np.array([0.1, 0.6, 0.3])
    rng = np.random.default_rng(1)
    low = [sample_next_token(probs, 0.2, rng) for _ in range(300)]
    high = [sample_next_token(probs, 1.5, rng) for _ in range(300)]
    assert low.count(1) > high.count(1)


def test_generate_token_ids_length_and_range(rng):
    ids = generate_token_ids(StubModel(), np.zeros(10, dtype=np.int32), 25, 1.0, rng)
    assert len(ids) == 25 and all(0 <= i < 5 for i in ids)


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError) as info:
        load_generation_assets(tmp_path / "models", tmp_path / "processed")
    assert "src.train" in info.value.fix


@pytest.fixture()
def tiny_trained_setup(processed_dir, tmp_path):
    """An untrained but correctly-shaped model saved like the real one."""
    vocab_size = len(json.loads((processed_dir / config.VOCAB_FILENAME).read_text())["tokens"])
    model = build_model(vocab_size, 10, embedding_dim=8, lstm_units=8)
    models = tmp_path / "models"
    models.mkdir()
    model.save(str(models / config.BEST_MODEL_FILENAME))
    (models / config.MODEL_CONFIG_FILENAME).write_text(
        json.dumps({"sequence_length": 10, "vocab_size": vocab_size})
    )
    return models, processed_dir


def test_generate_music_creates_midi_and_is_reproducible(tiny_trained_setup, tmp_path):
    models, processed = tiny_trained_setup
    out = tmp_path / "outputs"
    first = generate_music(30, 1.0, 11, models, processed, out, "a.mid")
    second = generate_music(30, 1.0, 11, models, processed, out, "b.mid")
    assert first.midi_path.exists() and first.midi_path.stat().st_size > 0
    assert first.num_events == 30
    assert first.midi_path.read_bytes() == second.midi_path.read_bytes()


def test_existing_file_is_not_overwritten(tiny_trained_setup, tmp_path):
    models, processed = tiny_trained_setup
    out = tmp_path / "outputs"
    a = generate_music(10, 1.0, 1, models, processed, out, "same.mid")
    b = generate_music(10, 1.0, 1, models, processed, out, "same.mid")
    assert a.midi_path != b.midi_path and a.midi_path.exists() and b.midi_path.exists()
