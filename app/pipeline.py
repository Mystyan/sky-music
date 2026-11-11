from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

from .config import settings
from .schemas import Quality, QUALITY_SPECS
from .stages.repo_arrange import arrange_from_midi
from .stages.export import to_abc15, to_skystudio
from .stages.harmony import detect_key, infer_chords
from .stages.instruments import transcribe_instruments
from .stages.rhythm import analyze as analyze_rhythm
from .stages.separation import separate
from .stages.vocal import transcribe_vocal
from .audio import convert_to_wav, ffprobe_duration

STAGES = ["ingest", "load", "separate", "vocal", "instruments", "rhythm_harmony", "arrange", "export"]


def stage_models(quality: Quality) -> dict[str, str]:
    """每一档、每个阶段实际使用的模型，供前端展示。"""
    spec = QUALITY_SPECS[quality]
    return {
        "ingest": "ffmpeg",
        "load": "librosa",
        "separate": spec["separator"],
        "vocal": f'{spec["vocal_model"]}-{spec["vocal_size"]}',
        "instruments": f'{spec["instrument_model"]}-{spec["instrument_size"]}',
        "rhythm_harmony": "librosa",
        "arrange": "sky-scores",
        "export": "skystudio+abc15",
    }


def run_pipeline(
    audio_path: str | Path,
    quality: Quality = Quality.medium,
    title: str = "未命名",
    author: str = "",
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    emit = on_event or (lambda _event: None)
    started = time.perf_counter()
    spec = QUALITY_SPECS[quality]
    models = stage_models(quality)
    audio_path = Path(audio_path)
    duration = ffprobe_duration(audio_path)
    if duration <= 0:
        raise ValueError("无法读取音频时长")
    if duration > settings.max_duration_sec:
        raise ValueError(f"音频 {duration:.1f}s 超过上限 {settings.max_duration_sec}s")

    stages: list[dict] = []
    warnings: list[str] = []

    def begin(name: str) -> None:
        emit({"type": "stage_start", "stage": name, "model": models[name]})

    def mark(name: str, mode: str, t0: float) -> None:
        item = {"stage": name, "mode": mode, "seconds": round(time.perf_counter() - t0, 2)}
        stages.append(item)
        emit({"type": "stage_done", **item})

    with __import__("tempfile").TemporaryDirectory(dir=str(settings.temp_dir)) as tmp:
        workdir = Path(tmp)
        begin("ingest")
        t0 = time.perf_counter()
        wav = convert_to_wav(audio_path, workdir / "input.wav", sample_rate=44100, mono=False)
        mark("ingest", models["ingest"], t0)

        begin("load")
        t0 = time.perf_counter()
        y22, sr22 = librosa.load(str(wav), sr=22050, mono=True)
        y16 = librosa.resample(y22, orig_sr=sr22, target_sr=16000)
        sr16 = 16000
        mark("load", models["load"], t0)

        begin("separate")
        t0 = time.perf_counter()
        stems = separate(wav, workdir, spec)
        warnings.extend(stems.warnings)
        mark("separate", stems.mode, t0)

        begin("vocal")
        t0 = time.perf_counter()
        melody, vocal_mode, vocal_warnings = transcribe_vocal(stems.vocals, duration, spec)
        warnings.extend(vocal_warnings)
        mark("vocal", vocal_mode, t0)

        begin("instruments")
        t0 = time.perf_counter()
        tracks, inst_mode, inst_warnings = transcribe_instruments(stems.instrumental, duration, spec)
        warnings.extend(inst_warnings)
        mark("instruments", inst_mode, t0)

        begin("rhythm_harmony")
        t0 = time.perf_counter()
        analysis = analyze_rhythm(y22, sr22, duration)
        analysis.key = "C" if settings.mock_mode else detect_key(y16, sr16)
        analysis.chords = infer_chords(tracks or [], analysis, duration)
        mark("rhythm_harmony", models["rhythm_harmony"], t0)

        begin("arrange")
        t0 = time.perf_counter()
        arrangement = arrange_from_midi(melody, tracks, analysis, quality, title, author)
        mark("arrange", models["arrange"], t0)

        begin("export")
        t0 = time.perf_counter()
        sky_json = to_skystudio(arrangement, title, author)
        abc15 = to_abc15(arrangement, title, author)
        mark("export", models["export"], t0)

    warnings = list(dict.fromkeys(warnings))
    return {
        "title": title,
        "author": author,
        "quality": quality.value,
        "duration": round(duration, 2),
        "bpm": round(arrangement.bpm, 2),
        "key": arrangement.key,
        "shift": arrangement.shift,
        "stats": arrangement.stats,
        "stages": stages,
        "stage_models": models,
        "warnings": warnings,
        "sky_json": sky_json,
        "abc15": abc15,
        "mode": "mock" if settings.mock_mode else "models",
        "elapsed": round(time.perf_counter() - started, 2),
    }
