import numpy as np
import pytest

from src import config
from src.dataset import (
    build_vocabulary,
    create_sequences,
    encode_tokens,
    load_processed,
    load_vocabulary,
    pick_seed_window,
    train_val_split,
)
from src.utils import DatasetError, ValidationError


def test_vocabulary_is_sorted_and_filters_rare_tokens():
    vocab = build_vocabulary({"D4": 5, "C4": 3, "E4": 1}, min_count=2)
    assert vocab == ["C4", "D4"]


def test_encode_tokens_drops_unknown():
    encoded = encode_tokens(["C4", "X9", "D4"], {"C4": 0, "D4": 1})
    assert encoded.tolist() == [0, 1]


def test_create_sequences_shapes_and_targets():
    ids = np.arange(30, dtype=np.int32)
    x, y = create_sequences(ids, np.array([30]), sequence_length=5)
    assert x.shape == (25, 5) and y.shape == (25,)
    assert x[0].tolist() == [0, 1, 2, 3, 4] and y[0] == 5
    assert y[-1] == 29


def test_sequences_never_cross_file_boundaries():
    ids = np.concatenate([np.zeros(10), np.ones(10) * 7]).astype(np.int32)
    x, y = create_sequences(ids, np.array([10, 10]), sequence_length=4)
    for row, target in zip(x, y):
        assert len(set(row.tolist() + [int(target)])) == 1


def test_max_sequences_caps_and_is_reproducible():
    ids = np.arange(200, dtype=np.int32)
    a = create_sequences(ids, np.array([200]), 5, max_sequences=20, seed=3)
    b = create_sequences(ids, np.array([200]), 5, max_sequences=20, seed=3)
    assert a[0].shape == (20, 5) and np.array_equal(a[0], b[0])


def test_too_little_data_gives_clear_error():
    with pytest.raises(DatasetError) as info:
        create_sequences(np.arange(10, dtype=np.int32), np.array([10]), 8, min_sequences=50)
    assert "more MIDI" in info.value.fix


def test_invalid_sequence_length_rejected():
    with pytest.raises(ValidationError):
        create_sequences(np.arange(10, dtype=np.int32), np.array([10]), 1)


def test_train_val_split_sizes():
    x = np.zeros((100, 5), dtype=np.int32)
    y = np.zeros(100, dtype=np.int32)
    xt, yt, xv, yv = train_val_split(x, y, 0.1, seed=1)
    assert len(xt) == 90 and len(xv) == 10
    with pytest.raises(ValidationError):
        train_val_split(x, y, 0.9, seed=1)


def test_processed_files_load_and_match(processed_dir):
    ids, lengths = load_processed(processed_dir / config.PROCESSED_DATA_FILENAME)
    vocab = load_vocabulary(processed_dir / config.VOCAB_FILENAME)
    assert ids.sum() >= 0 and int(lengths.sum()) == ids.size
    assert int(ids.max()) < len(vocab)


def test_pick_seed_window_shape_and_reproducible(processed_dir):
    ids, lengths = load_processed(processed_dir / config.PROCESSED_DATA_FILENAME)
    a = pick_seed_window(ids, lengths, 20, np.random.default_rng(5))
    b = pick_seed_window(ids, lengths, 20, np.random.default_rng(5))
    assert a.shape == (20,) and np.array_equal(a, b)
