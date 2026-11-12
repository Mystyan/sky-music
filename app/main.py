from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from . import tasks
from .audio import download_file
from .config import settings
from .errors import ModelUnavailable
from .pipeline import STAGES, run_pipeline, stage_models
from .schemas import Quality, QUALITY_SPECS, TranscribeUrlRequest

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="sky-music", version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/info")
def root() -> dict:
    return {
        "service": "sky-music",
        "version": settings.app_version,
        "mode": "mock" if settings.mock_mode else "models",
        "endpoints": [
            "/health", "/api/status", "/api/qualities",
            "/api/tasks", "/api/tasks/{task_id}",
            "/api/transcribe", "/api/transcribe-url",
        ],
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


@app.get("/api/status")
def status() -> dict:
    """服务状态：是否就绪、GPU、当前在跑的任务数、各档位每阶段模型。"""
    load = tasks.stats()
    return {
        "ok": True,
        "ready": True,
        "device": settings.device,
        "mode": "mock" if settings.mock_mode else "models",
        "stages": STAGES,
        "busy": load["running"],
        "queued": max(0, load["tasks"] - load["running"]),
        "qualities": {
            quality.value: {
                "label": spec["label"],
                "description": spec["description"],
                "stage_models": stage_models(quality),
            }
            for quality, spec in QUALITY_SPECS.items()
        },
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
            "stage_models": stage_models(quality),
        }
        for quality, spec in QUALITY_SPECS.items()
    ]


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(settings.temp_dir)) as fh:
        path = Path(fh.name)
        total = 0
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                fh.close()
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="文件过大")
            fh.write(chunk)
    if total == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="空文件")
    return path


@app.post("/api/tasks")
def create_task(
    file: UploadFile = File(...),
    quality: Quality = Form(Quality.low),
    title: str = Form("未命名"),
    author: str = Form(""),
) -> dict:
    """创建转谱任务，立即返回 task_id；用 GET /api/tasks/{id} 查询进度。"""
    path = _save_upload(file)
    task_id = tasks.create(path, quality, title, author)
    return {"task_id": task_id, "status_url": f"/api/tasks/{task_id}"}


@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str) -> dict:
    data = tasks.snapshot(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="任务不存在或已过期（实例重启后任务会丢失）")
    return data


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
