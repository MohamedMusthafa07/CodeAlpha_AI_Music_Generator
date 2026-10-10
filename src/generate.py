"""Generate music with the trained LSTM and export a MIDI file.

Run from the project root (after training):
    python -m src.generate
    python -m src.generate --notes 300 --temperature 0.8 --seed 7
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src import config
from src.dataset import load_processed, load_vocabulary, pick_seed_window
from src.midi_utils import (
    estimate_duration_seconds,
    tokens_to_stream,
    write_midi_file,
)
from src.utils import (
    GenerationError,
    MusicGenError,
    ModelNotFoundError,
    PathLike,
    ValidationError,
    load_json,
    setup_logging,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationAssets:
    """Everything needed to generate: model, vocabulary and seed data."""

    model: object
    vocabulary: list[str]
    sequence_length: int
    ids: np.ndarray
    lengths: np.ndarray
    model_file: str


@dataclass
class GenerationResult:
    """Information about one generated composition."""

    midi_path: Path
    num_events: int
    duration_seconds: float
    temperature: float
    seed: Optional[int]
    model_name: str = "LSTM"


# --------------------------------------------------------------------------
# Validation and sampling
# --------------------------------------------------------------------------
def validate_temperature(
    temperature: float, cfg: config.GenerationConfig = config.DEFAULT_GENERATION
) -> float:
    """Ensure the temperature is a number in the allowed range."""
    try:
        value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Temperature must be a number.", fix="Use a value like 1.0.") from exc
    if not np.isfinite(value) or not cfg.min_temperature <= value <= cfg.max_temperature:
        raise ValidationError(
            f"Temperature {temperature} is not allowed.",
            reason=f"It must be between {cfg.min_temperature} and {cfg.max_temperature}.",
            fix="Use 0.5 (predictable), 1.0 (balanced) or 1.5 (experimental).",
        )
    return value


def validate_num_notes(num_notes: int, cfg: config.GenerationConfig = config.DEFAULT_GENERATION) -> int:
    """Ensure the requested number of notes is a sensible whole number."""
    if isinstance(num_notes, bool) or int(num_notes) != num_notes:
        raise ValidationError("Number of notes must be a whole number.", fix="Use a value like 200.")
    if not cfg.min_notes <= int(num_notes) <= cfg.max_notes:
        raise ValidationError(
            f"Number of notes {num_notes} is not allowed.",
            reason=f"It must be between {cfg.min_notes} and {cfg.max_notes}.",
            fix="Use a value like 200.",
        )
    return int(num_notes)


def sample_next_token(probabilities: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Pick the next token ID using temperature-scaled probabilistic sampling."""
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
    logits = np.log(probs) / temperature
    logits -= logits.max()  # numerical stability
    scaled = np.exp(logits)
    scaled /= scaled.sum()
    return int(rng.choice(len(scaled), p=scaled))


def _make_predict_function(model):
    """Return a fast function that maps an input window to next-token probabilities.

    Real Keras models are wrapped in tf.function (much faster than calling the
    model eagerly 200+ times). Any other callable (e.g. a test stub) is used as is.
    """
    try:
        import tensorflow as tf

        if isinstance(model, tf.keras.Model):
            compiled = tf.function(lambda batch: model(batch, training=False), reduce_retracing=True)
            return lambda window: compiled(tf.constant(window)).numpy()
    except ImportError:  # TensorFlow missing: fall back to a plain call
        pass
    return lambda window: model(window)


def generate_token_ids(
    model,
    seed_ids: np.ndarray,
    num_notes: int,
    temperature: float,
    rng: np.random.Generator,
) -> list[int]:
    """Repeatedly predict the next token and slide the input window forward."""
    predict = _make_predict_function(model)
    window = np.array(seed_ids, dtype=np.int32).reshape(1, -1)
    generated: list[int] = []
    for _ in range(num_notes):
        probs = np.asarray(predict(window))[0]
        next_id = sample_next_token(probs, temperature, rng)
        generated.append(next_id)
        window = np.concatenate([window[:, 1:], [[next_id]]], axis=1).astype(np.int32)
    return generated


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def model_files_status(
    model_dir: PathLike = config.MODEL_DIR, processed_dir: PathLike = config.PROCESSED_DIR
) -> dict:
    """Report which required files exist (used by the CLI and the Streamlit app)."""
    model_dir, processed_dir = Path(model_dir), Path(processed_dir)
    model_file = next(
        (
            model_dir / name
            for name in (config.BEST_MODEL_FILENAME, config.FINAL_MODEL_FILENAME)
            if (model_dir / name).exists()
        ),
        None,
    )
    return {
        "model_file": model_file,
        "model_config": (model_dir / config.MODEL_CONFIG_FILENAME).exists(),
        "vocabulary": (processed_dir / config.VOCAB_FILENAME).exists(),
        "processed": (processed_dir / config.PROCESSED_DATA_FILENAME).exists(),
    }


def load_generation_assets(
    model_dir: PathLike = config.MODEL_DIR, processed_dir: PathLike = config.PROCESSED_DIR
) -> GenerationAssets:
    """Load and cross-check the model, vocabulary and seed data."""
    status = model_files_status(model_dir, processed_dir)
    if status["model_file"] is None or not status["model_config"]:
        raise ModelNotFoundError(
            "No trained model was found.",
            reason="Training has not been completed yet.",
            fix="Run preprocessing and training first: python -m src.preprocessing, then python -m src.train",
        )
    if not status["vocabulary"] or not status["processed"]:
        raise ModelNotFoundError(
            "The vocabulary or processed data is missing.",
            reason="Generation must use exactly the vocabulary created during training.",
            fix="Run: python -m src.preprocessing, then python -m src.train",
        )

    from tensorflow import keras

    model_config = load_json(Path(model_dir) / config.MODEL_CONFIG_FILENAME)
    vocabulary = load_vocabulary(Path(processed_dir) / config.VOCAB_FILENAME)
    ids, lengths = load_processed(Path(processed_dir) / config.PROCESSED_DATA_FILENAME)

    try:
        model = keras.models.load_model(str(status["model_file"]), compile=False)
    except Exception as exc:  # Keras raises many different exception types on load
        logger.exception("Model loading failed")
        raise ModelNotFoundError(
            "The trained model could not be loaded.",
            reason=f"Keras reported: {exc}",
            fix="Train the model again: python -m src.train",
        ) from exc

    sequence_length = int(model_config.get("sequence_length", 0))
    if (
        model.output_shape[-1] != len(vocabulary)
        or model.input_shape[1] != sequence_length
        or int(model_config.get("vocab_size", -1)) != len(vocabulary)
    ):
        raise ModelNotFoundError(
            "The model and vocabulary do not match.",
            reason="They were created from different preprocessing or training runs.",
            fix="Run preprocessing and training again so both are created together.",
        )
    return GenerationAssets(model, vocabulary, sequence_length, ids, lengths, Path(status["model_file"]).name)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def generate_music(
    num_notes: int = config.DEFAULT_GENERATION.num_notes,
    temperature: float = config.DEFAULT_GENERATION.temperature,
    seed: Optional[int] = config.DEFAULT_GENERATION.random_seed,
    model_dir: PathLike = config.MODEL_DIR,
    processed_dir: PathLike = config.PROCESSED_DIR,
    output_dir: PathLike = config.OUTPUT_DIR,
    output_name: Optional[str] = None,
    overwrite: bool = False,
    assets: Optional[GenerationAssets] = None,
    cfg: config.GenerationConfig = config.DEFAULT_GENERATION,
) -> GenerationResult:
    """Generate a composition and save it as a MIDI file."""
    temperature = validate_temperature(temperature, cfg)
    num_notes = validate_num_notes(num_notes, cfg)
    if seed is not None and (isinstance(seed, bool) or int(seed) != seed or seed < 0):
        raise ValidationError("Random seed must be a non-negative whole number.", fix="Use a value like 42.")

    if assets is None:
        assets = load_generation_assets(model_dir, processed_dir)

    rng = np.random.default_rng(seed)
    seed_ids = pick_seed_window(assets.ids, assets.lengths, assets.sequence_length, rng)
    logger.info("Generating %d notes (temperature=%s, seed=%s)", num_notes, temperature, seed)
    try:
        new_ids = generate_token_ids(assets.model, seed_ids, num_notes, temperature, rng)
    except MusicGenError:
        raise
    except Exception as exc:
        logger.exception("Generation failed")
        raise GenerationError(
            "Music generation failed.",
            reason=f"TensorFlow reported: {exc}",
            fix="Retrain the model with python -m src.train and try again.",
        ) from exc

    tokens = [assets.vocabulary[i] for i in new_ids]
    score, used = tokens_to_stream(tokens, cfg.note_step, cfg.tempo_bpm)
    path = write_midi_file(score, output_dir, output_name, overwrite)
    result = GenerationResult(
        midi_path=path,
        num_events=used,
        duration_seconds=estimate_duration_seconds(used, cfg.note_step, cfg.tempo_bpm),
        temperature=temperature,
        seed=seed,
    )
    logger.info("Saved MIDI: %s", path)
    return result


def main(argv: list[str] | None = None) -> int:
    defaults = config.DEFAULT_GENERATION
    parser = argparse.ArgumentParser(description="Generate a MIDI file with the trained LSTM.")
    parser.add_argument("--notes", type=int, default=defaults.num_notes, help="Number of notes/chords")
    parser.add_argument("--temperature", type=float, default=defaults.temperature, help="0.5 safe .. 1.5 experimental")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible output")
    parser.add_argument("--output", default=None, help="Output file name (saved inside outputs/)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists")
    parser.add_argument("--open", action="store_true", help="Open the MIDI in your default player (Windows)")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        result = generate_music(
            args.notes, args.temperature, args.seed, output_name=args.output, overwrite=args.overwrite
        )
    except MusicGenError as exc:
        logger.error(exc.message)
        print("\n" + exc.describe())
        return 1

    print("\nGeneration successful.")
    print(f"  MIDI file : {result.midi_path}")
    print(f"  Events    : {result.num_events}")
    print(f"  Duration  : about {result.duration_seconds:.0f} seconds")
    print("Open the file with any MIDI-compatible player (playback depends on your system).")
    if args.open:
        if hasattr(os, "startfile"):
            os.startfile(str(result.midi_path))  # type: ignore[attr-defined]  # Windows only
        else:
            print("--open is only supported on Windows; open the file manually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
