"""Writable data locations for source, installed packages, and frozen apps."""
import os
from pathlib import Path
import sys


def project_root():
    return Path(__file__).resolve().parents[1]


def get_data_root():
    override = os.environ.get("DEADLINE_ENGINE_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        local = os.environ.get("XDG_DATA_HOME")
        base = Path(local) if local else Path.home() / ".local" / "share"
    return base / "DeadlineInference"


def get_cache_root():
    if os.environ.get("DEADLINE_ENGINE_DATA_DIR") or getattr(sys, "frozen", False):
        return get_data_root() / "cache"
    root = project_root()
    # Editable/source checkouts keep their existing explicit weight cache.
    if (root / "scripts" / "setup.ps1").is_file():
        return root / ".cache"
    return get_data_root() / "cache"
