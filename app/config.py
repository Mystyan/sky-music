from __future__ import annotations

import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _detect_device() -> str:
    explicit = os.getenv("DEVICE")
    if explicit:
        return explicit
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        return "cuda"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Settings:
    # Mock mode is for tests only. Production always requires real models.
    mock_mode: bool = _bool("MOCK_MODE", False)
    device: str = _detect_device()
    model_dir: Path = Path(os.getenv("MODEL_DIR", "/mnt/models"))
    max_duration_sec: int = int(os.getenv("MAX_DURATION_SEC", "600"))
    temp_dir: Path = Path(os.getenv("TEMP_DIR", "/tmp/sky-music"))
    hf_token: str | None = os.getenv("HF_TOKEN") or None

    # Optional model locations. Empty means "unavailable, fall back".
    subroformer_dir: str = os.getenv("SUBROFORMER_DIR", "")
    vocalparse_ckpt: str = os.getenv("VOCALPARSE_CKPT", "")
    game_dir: str = os.getenv("GAME_DIR", "")

    app_version: str = os.getenv("APP_VERSION", "0.1.0")

    def ensure_dirs(self) -> None:
        self.temp_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
