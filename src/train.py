"""Train the LSTM model.

Run from the project root (after preprocessing):
    python -m src.train
    python -m src.train --epochs 30 --batch-size 128
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src import config
from src.dataset import (
    create_sequences,
    load_processed,
    load_vocabulary,
    train_val_split,
    validate_sequence_length,
)
from src.utils import (
    MusicGenError,
    PathLike,
    TrainingError,
    ensure_dir,
    save_json,
    set_global_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)

# progress(epoch_number, total_epochs, metrics_dict)
TrainProgress = Optional[Callable[[int, int, dict], None]]


def _validate_config(cfg: config.TrainingConfig) -> None:
    validate_sequence_length(cfg.sequence_length)
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise TrainingError(
            "Epochs and batch size must be at least 1.",
            fix="Use values such as epochs=20 and batch_size=64.",
        )
    if not 0.0 < cfg.learning_rate < 1.0:
        raise TrainingError("Learning rate must be between 0 and 1.", fix="Use a value such as 0.001.")
    if not 0.0 <= cfg.dropout < 1.0:
        raise TrainingError("Dropout must be at least 0 and below 1.", fix="Use a value such as 0.3.")


def _save_history_plot(history: dict, path: Path) -> None:
    """Save a loss/accuracy chart. A plotting problem must never kill training."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")):
            ax.plot(history.get(key, []), label="train")
            ax.plot(history.get(f"val_{key}", []), label="validation")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as exc:  # plotting is optional; log and continue
        logger.warning("Could not save training plot: %s", exc)


def train_model(
    cfg: config.TrainingConfig = config.DEFAULT_TRAINING,
    processed_dir: PathLike = config.PROCESSED_DIR,
    model_dir: PathLike = config.MODEL_DIR,
    progress: TrainProgress = None,
) -> dict:
    """Train the model and return a summary dictionary with real metrics."""
    _validate_config(cfg)
    processed = Path(processed_dir)
    out_dir = ensure_dir(model_dir)

    # Heavy imports happen here so the rest of the app starts quickly.
    from tensorflow import keras

    from src.model import build_model, validate_shapes

    logger.info("Loading processed data from %s", processed)
    vocabulary = load_vocabulary(processed / config.VOCAB_FILENAME)
    ids, lengths = load_processed(processed / config.PROCESSED_DATA_FILENAME)
    vocab_size = len(vocabulary)
    if int(ids.max()) >= vocab_size:
        raise TrainingError(
            "Processed data and vocabulary do not match.",
            fix="Run preprocessing again: python -m src.preprocessing",
        )

    x, y = create_sequences(
        ids, lengths, cfg.sequence_length,
        max_sequences=cfg.max_sequences, seed=cfg.random_seed, min_sequences=cfg.min_sequences,
    )
    x_train, y_train, x_val, y_val = train_val_split(x, y, cfg.validation_split, cfg.random_seed)
    logger.info("Sequences: %d train / %d validation | vocabulary %d", len(x_train), len(x_val), vocab_size)

    keras.utils.set_random_seed(cfg.random_seed)
    set_global_seed(cfg.random_seed)
    model = build_model(
        vocab_size, cfg.sequence_length, cfg.embedding_dim, cfg.lstm_units, cfg.dropout, cfg.learning_rate
    )
    validate_shapes(model, x_train, y_train, vocab_size)
    validate_shapes(model, x_val, y_val, vocab_size)

    best_path = out_dir / config.BEST_MODEL_FILENAME
    final_path = out_dir / config.FINAL_MODEL_FILENAME
    save_json(
        out_dir / config.MODEL_CONFIG_FILENAME,
        {
            "sequence_length": cfg.sequence_length,
            "vocab_size": vocab_size,
            "embedding_dim": cfg.embedding_dim,
            "lstm_units": cfg.lstm_units,
            "dropout": cfg.dropout,
            "created": datetime.now().isoformat(timespec="seconds"),
        },
    )

    class _Progress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):  # noqa: D401
            metrics = {k: float(v) for k, v in (logs or {}).items()}
            logger.info("Epoch %d/%d - %s", epoch + 1, cfg.epochs, metrics)
            if progress:
                progress(epoch + 1, cfg.epochs, metrics)

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.early_stopping_patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=cfg.lr_reduce_patience),
        _Progress(),
    ]

    try:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            callbacks=callbacks, verbose=2,
        )
    except MusicGenError:
        raise
    except Exception as exc:  # TensorFlow raises many different exception types
        logger.exception("Training failed")
        raise TrainingError(
            "Training stopped because of an error.",
            reason=f"TensorFlow reported: {exc}",
            fix="Try a smaller batch size or sequence length, and check logs/ for details.",
        ) from exc

    try:
        model.save(str(final_path))
    except Exception as exc:
        raise TrainingError(
            "The trained model could not be saved.",
            reason=f"Saving failed: {exc}",
            fix="Check free disk space and that the models/ folder is writable.",
        ) from exc

    hist = {k: [float(v) for v in values] for k, values in history.history.items()}
    save_json(out_dir / config.HISTORY_FILENAME, hist)
    _save_history_plot(hist, out_dir / config.HISTORY_PLOT_FILENAME)

    summary = {
        "completed": datetime.now().isoformat(timespec="seconds"),
        "epochs_run": len(hist.get("loss", [])),
        "epochs_requested": cfg.epochs,
        "best_val_loss": min(hist["val_loss"]),
        "final_train_loss": hist["loss"][-1],
        "final_train_accuracy": hist["accuracy"][-1],
        "best_val_accuracy": max(hist["val_accuracy"]),
        "train_sequences": int(len(x_train)),
        "validation_sequences": int(len(x_val)),
        "vocab_size": vocab_size,
        "best_model": best_path.name,
        "final_model": final_path.name,
        "config": asdict(cfg),
    }
    save_json(out_dir / config.SUMMARY_FILENAME, summary)
    logger.info("Training complete. Best val loss %.4f", summary["best_val_loss"])
    return summary


def main(argv: list[str] | None = None) -> int:
    defaults = config.DEFAULT_TRAINING
    parser = argparse.ArgumentParser(description="Train the LSTM music model.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--sequence-length", type=int, default=defaults.sequence_length)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--validation-split", type=float, default=defaults.validation_split)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = replace(
        defaults,
        epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.sequence_length,
        learning_rate=args.learning_rate, validation_split=args.validation_split, random_seed=args.seed,
    )
    try:
        summary = train_model(cfg)
    except MusicGenError as exc:
        logger.error(exc.message)
        print("\n" + exc.describe())
        return 1
    print("\nTraining completed.")
    print(f"  Epochs run         : {summary['epochs_run']}")
    print(f"  Best validation loss: {summary['best_val_loss']:.4f}")
    print(f"  Final training loss : {summary['final_train_loss']:.4f}")
    print(f"  Model saved to      : models/{summary['best_model']}")
    print("Next step: python -m src.generate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
