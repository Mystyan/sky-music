from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import httpx


def ffprobe_duration(path: str | Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ValueError(f"ffprobe failed: {proc.stderr[-300:]}")
    data = json.loads(proc.stdout or "{}")
    return float(data.get("format", {}).get("duration", 0.0))


def convert_to_wav(src: str | Path, dst: str | Path, sample_rate: int = 44100, mono: bool = False) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", str(sample_rate)]
    if mono:
        cmd += ["-ac", "1"]
    else:
        cmd += ["-ac", "2"]
    cmd += ["-vn", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dst.exists():
        raise ValueError(f"ffmpeg failed: {proc.stderr[-300:]}")
    return dst


def copy_as_wav(src: str | Path, dst: str | Path) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


async def download_file(url: str, dst: str | Path, timeout: float = 120.0) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with dst.open("wb") as fh:
                async for chunk in response.aiter_bytes():
                    fh.write(chunk)
    return dst


def run_in_thread(func, *args, **kwargs):
    """Small helper for sync model calls inside async handlers."""
    return asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))
