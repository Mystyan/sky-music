from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import settings
from ..errors import ModelUnavailable


@dataclass
class Stems:
    vocals: Path
    instrumental: Path
    mode: str
    warnings: list[str] = field(default_factory=list)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] <= 2:
        array = array.T
    sf.write(str(path), array, sr)
    return path


def _subroformer(src: Path, workdir: Path) -> Stems:
    from subroformer.inference.engine import (  # type: ignore
        load_audio,
        load_separator,
        resolve_amp_dtype,
        separate_audio,
    )

    kwargs = {"device": settings.device}
    if settings.subroformer_dir:
        kwargs["checkpoint"] = settings.subroformer_dir
    try:
        model, config = load_separator(**kwargs)
    except TypeError:
        model, config = load_separator(device=settings.device)

    sr = int(config.audio.sample_rate)
    audio = load_audio(str(src), sr)
    stems = separate_audio(
        model,
        audio,
        config,
        amp_dtype=resolve_amp_dtype(next(model.parameters()).device),
    )
    vocals = _write_wav(workdir / "vocals.wav", stems["vocals"], sr)
    instrumental = stems["bass"] + stems["drums"] + stems["other"]
    inst = _write_wav(workdir / "instrumental.wav", instrumental, sr)
    return Stems(vocals=vocals, instrumental=inst, mode="subroformer")


def _demucs(src: Path, workdir: Path) -> Stems:
    from demucs.api import Separator  # type: ignore

    separator = Separator(device=settings.device)
    _, separated = separator.separate_audio_file(str(src))
    vocals = separated["vocals"].cpu().numpy()
    inst = sum(separated[k].cpu().numpy() for k in separated if k != "vocals")
    sr = separator.samplerate
    return Stems(
        vocals=_write_wav(workdir / "vocals.wav", vocals, sr),
        instrumental=_write_wav(workdir / "instrumental.wav", inst, sr),
        mode="demucs",
    )


def separate(src: str | Path, workdir: str | Path, spec: dict) -> Stems:
    src = Path(src)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if settings.mock_mode or spec.get("separator") == "none":
        return Stems(vocals=src, instrumental=src, mode="mock",
                     warnings=["MOCK_MODE：未执行真实分轨。"])

    warnings: list[str] = []
    preferred = spec.get("separator", "auto")

    if preferred in ("auto", "subroformer"):
        try:
            return _subroformer(src, workdir)
        except Exception as exc:
            warnings.append(f"SubRoformer 不可用：{exc.__class__.__name__}: {exc}")

    if preferred in ("auto", "demucs", "subroformer"):
        try:
            return _demucs(src, workdir)
        except Exception as exc:
            warnings.append(f"Demucs 不可用：{exc.__class__.__name__}: {exc}")

    raise ModelUnavailable("分轨模型不可用：" + "；".join(warnings))
