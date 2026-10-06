"""JWT secret resolution shared by user and agent auth."""

from __future__ import annotations

import os

from mas.config import settings_config


def jwt_secret() -> str:
    cfg = settings_config()["jwt"]
    return os.environ.get(cfg["secret_env"], cfg["dev_secret"])
