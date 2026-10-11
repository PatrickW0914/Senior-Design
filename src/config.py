"""config.py - load config.yaml and resolve repo-relative paths."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def repo_path(p: str | Path) -> Path:
    """Relative paths in the config are relative to the repo root, not the working directory."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
