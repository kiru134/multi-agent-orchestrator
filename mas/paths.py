"""Central path constants so every module agrees on where the repo root is."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
SEED_DIR = DATA_DIR / "seed"
PROMPTS_DIR = CONFIG_DIR / "prompts"
