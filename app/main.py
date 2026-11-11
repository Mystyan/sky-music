from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .audio import download_file
from .config import settings
from .errors import ModelUnavailable
from .pipeline import run_pipeline
from .schemas import Quality, QUALITY_SPECS, TranscribeUrlRequest

app = FastAPI(title="sky-music", version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))


@app.get("/")
def root() -> dict:
    return {
        "service": "sky-music",
        "version": settings.app_version,
        "mode": "mock" if settings.mock_mode else "models",
        "endpoints": ["/health", "/api/qualities", "/api/transcribe", "/api/transcribe-url"],
    }


@app.get("/health")
@app.get("/healthz")
def health() -> dict:
    return {
        "ok": True,
        "version": settings.app_version,
        "mode": "mock" if settings.mock_mode else "models",
        "device": settings.device,
        "qualities": [q.value for q in Quality],
    }


@app.get("/api/qualities")
def qualities() -> list[dict]:
    return [
        {
            "id": quality.value,
            "label": spec["label"],
            "description": spec["description"],
            "vocal_model": f'{spec["vocal_model"]}-{spec["vocal_size"]}',
            "instrument_model": f'{spec["instrument_model"]}-{spec["instrument_size"]}',
            "separator": spec["separator"],
        }
        for quality, spec in QUALITY_SPECS.items()
    ]


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    quality: Quality = Form(Quality.medium),
    title: str = Form("未命名"),
    author: str = Form(""),
) -> dict:
    suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(settings.temp_dir)) as fh:
            tmp_path = Path(fh.name)
            total = 0
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="文件过大")
                fh.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="空文件")
        return await asyncio.to_thread(run_pipeline, tmp_path, quality, title, author)
    except HTTPException:
        raise
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/api/transcribe-url")
async def transcribe_url(body: TranscribeUrlRequest) -> dict:
    tmp_path = None
    try:
        suffix = Path(body.audio_url.split("?")[0]).suffix or ".mp3"
        fd, raw_path = tempfile.mkstemp(suffix=suffix, dir=str(settings.temp_dir))
        os.close(fd)
        tmp_path = Path(raw_path)
        await download_file(body.audio_url, tmp_path)
        return await asyncio.to_thread(
            run_pipeline, tmp_path, body.quality, body.title, body.author
        )
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
