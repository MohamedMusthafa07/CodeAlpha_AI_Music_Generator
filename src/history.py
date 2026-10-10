"""Persistent history of what you did in the app (generations, training, preprocessing).

The history is stored as a small JSON file next to your generated music:
    outputs/history.json
Deleting the history never deletes your MIDI files.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src import config
from src.utils import MusicGenError, PathLike, load_json, save_json

logger = logging.getLogger(__name__)

HISTORY_FILENAME = "history.json"
MAX_ENTRIES = 200
KINDS = ("generation", "training", "preprocessing")


def history_path(output_dir: PathLike = config.OUTPUT_DIR) -> Path:
    """Where the history file lives."""
    return Path(output_dir) / HISTORY_FILENAME


def load_history(path: Optional[PathLike] = None) -> list[dict[str, Any]]:
    """Return all saved entries, newest first. A missing or damaged file gives an empty list."""
    target = Path(path) if path else history_path()
    if not target.exists():
        return []
    try:
        data = load_json(target)
    except MusicGenError as exc:
        logger.warning("History file could not be read, starting fresh: %s", exc.message)
        return []
    if not isinstance(data, list):
        logger.warning("History file has an unexpected format, ignoring it.")
        return []
    entries = [item for item in data if isinstance(item, dict) and item.get("kind") in KINDS]
    return sorted(entries, key=lambda item: str(item.get("time", "")), reverse=True)


def add_entry(kind: str, details: dict[str, Any], path: Optional[PathLike] = None) -> dict[str, Any]:
    """Add one entry (kept to the newest MAX_ENTRIES) and return it."""
    if kind not in KINDS:
        raise ValueError(f"Unknown history kind '{kind}'. Use one of {KINDS}.")
    target = Path(path) if path else history_path()
    entry = {
        "id": uuid.uuid4().hex[:8],
        "kind": kind,
        "time": datetime.now().isoformat(timespec="seconds"),
        **details,
    }
    entries = load_history(target)
    entries.insert(0, entry)
    save_json(target, entries[:MAX_ENTRIES])
    return entry


def clear_history(path: Optional[PathLike] = None) -> None:
    """Remove all history entries (MIDI files are kept)."""
    target = Path(path) if path else history_path()
    save_json(target, [])


def untracked_midi_files(
    entries: list[dict[str, Any]], output_dir: PathLike = config.OUTPUT_DIR
) -> list[Path]:
    """MIDI files in outputs/ that are not in the history (e.g. made with the command line)."""
    folder = Path(output_dir)
    if not folder.is_dir():
        return []
    known = {entry.get("file") for entry in entries if entry.get("kind") == "generation"}
    files = [p for p in folder.glob("*.mid") if p.name not in known]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
