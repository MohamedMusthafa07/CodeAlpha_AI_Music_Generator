"""Shared, lightweight fixtures (tiny synthetic MIDI data; no long training)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.preprocessing import preprocess_dataset
from src.sample_data import create_sample_dataset

SMALL_PREPROCESS = config.PreprocessConfig(min_token_count=1, min_tokens_per_file=8)


@pytest.fixture()
def raw_dir(tmp_path):
    """A folder with 4 small valid MIDI files."""
    folder = tmp_path / "raw"
    create_sample_dataset(folder, files=4, length=120, seed=1)
    return folder


@pytest.fixture()
def processed_dir(tmp_path, raw_dir):
    """Preprocessed output for the fixture dataset."""
    out = tmp_path / "processed"
    preprocess_dataset(raw_dir, out, SMALL_PREPROCESS)
    return out


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
