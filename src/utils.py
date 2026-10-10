"""Shared helpers: friendly exceptions, logging, JSON and seeding."""

from __future__ import annotations

import json
import logging
import os
import random
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Union

import numpy as np

from src import config

PathLike = Union[str, Path]


class MusicGenError(Exception):
    """Base error carrying WHAT happened, WHY it happened and HOW to fix it."""

    def __init__(self, message: str, reason: str = "", fix: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.fix = fix

    def describe(self) -> str:
        """Return a readable multi-line description of the problem."""
        lines = [f"What happened: {self.message}"]
        if self.reason:
            lines.append(f"Why: {self.reason}")
        if self.fix:
            lines.append(f"How to fix: {self.fix}")
        return "\n".join(lines)


class DatasetError(MusicGenError):
    """Problems with the MIDI dataset or processed data."""


class MidiParseError(MusicGenError):
    """A single MIDI file could not be parsed."""


class ModelNotFoundError(MusicGenError):
    """The trained model or vocabulary is missing or incompatible."""


class TrainingError(MusicGenError):
    """Training could not be completed."""


class GenerationError(MusicGenError):
    """Music generation or MIDI export failed."""


class ValidationError(MusicGenError):
    """A user-supplied value is outside its allowed range."""


_LOGGER_NAME = "src"


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging once (console + rotating file) and return the logger."""
    logger = logging.getLogger(_LOGGER_NAME)
    if getattr(logger, "_musicgen_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    target_dir = Path(log_dir) if log_dir else config.LOG_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            target_dir / config.LOG_FILENAME,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not create log file in %s (%s). Logging to console only.", target_dir, exc)

    logger._musicgen_configured = True  # type: ignore[attr-defined]
    return logger


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MusicGenError(
            f"Could not create the folder '{directory}'.",
            reason=f"The operating system refused: {exc}",
            fix="Check that the project folder is writable and not locked by another program.",
        ) from exc
    return directory


def save_json(path: PathLike, data: Any) -> Path:
    """Write data as JSON (via a temporary file so a crash cannot corrupt it)."""
    target = Path(path)
    ensure_dir(target.parent)
    temp = target.with_suffix(target.suffix + ".tmp")
    try:
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp, target)
    except OSError as exc:
        raise MusicGenError(
            f"Could not write '{target.name}'.",
            reason=f"File error: {exc}",
            fix="Make sure the file is not open in another program and the folder is writable.",
        ) from exc
    return target


def load_json(path: PathLike) -> Any:
    """Read a JSON file, raising a friendly error if it is missing or damaged."""
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MusicGenError(
            f"The file '{source.name}' was not found.",
            reason="The step that creates it has not been run yet.",
            fix="Run preprocessing (python -m src.preprocessing) and training (python -m src.train).",
        ) from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise MusicGenError(
            f"The file '{source.name}' could not be read.",
            reason=f"It is damaged or unreadable: {exc}",
            fix="Delete the file and run the step that creates it again.",
        ) from exc


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
