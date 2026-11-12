from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def model_root() -> Path:
    root = Path(os.getenv("MODEL_DIR", "/tmp/models"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def hf_home() -> Path:
    home = Path(os.getenv("HF_HOME", str(model_root() / "hf")))
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(home)
    return home


def vocalparse_dir(report=None) -> Path:
    """VocalParse-1.7B 权重（约 4.1GB），首次调用时下载到 MODEL_DIR。"""
    target = Path(os.getenv("VOCALPARSE_CKPT") or (model_root() / "vocalparse"))
    if (target / "config.json").exists() and any(target.glob("*.safetensors")):
        return target
    if report:
        report("下载 VocalParse-1.7B 权重（约 4.1GB，仅首次）")
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="pymaster/VocalParse",
        local_dir=str(target),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"],
    )
    return target


def subroformer_checkpoint() -> str | None:
    """SubRoformer 默认由官方推理代码从 HuggingFace 自动拉取，这里只处理本地覆盖。"""
    override = os.getenv("SUBROFORMER_DIR")
    return override or None


def prepare(quality_value: str, report=None) -> list[str]:
    """按档位预取权重，返回提示信息。"""
    notes: list[str] = []
    hf_home()
    if quality_value == "high":
        vocalparse_dir(report=report)
        notes.append("VocalParse 权重就绪")
    return notes
