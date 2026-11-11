from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from .pipeline import STAGES, run_pipeline, stage_models
from .schemas import Quality

TASK_TTL_SEC = 3600
MAX_TASKS = 32

_lock = threading.Lock()
_tasks: dict[str, dict] = {}


def _now() -> float:
    return time.time()


def _gc_locked() -> None:
    now = _now()
    for key in [k for k, t in _tasks.items() if now - t["updated"] > TASK_TTL_SEC]:
        _tasks.pop(key, None)
    while len(_tasks) > MAX_TASKS:
        oldest = min(_tasks.items(), key=lambda kv: kv[1]["updated"])[0]
        _tasks.pop(oldest, None)


def snapshot(task_id: str) -> dict | None:
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return None
        return {
            "task_id": task_id,
            "status": task["status"],
            "quality": task["quality"],
            "stage": task["stage"],
            "stage_index": task["stage_index"],
            "total_stages": len(STAGES),
            "progress": round(task["stage_index"] / len(STAGES), 3),
            "models": task["models"],
            "events": list(task["events"]),
            "error": task["error"],
            "created_at": task["created"],
            "elapsed": round(_now() - task["created"], 1),
            "result": task["result"],
        }


def create(
    audio_path: Path,
    quality: Quality,
    title: str,
    author: str,
) -> str:
    task_id = uuid.uuid4().hex[:12]
    record = {
        "status": "running",
        "quality": quality.value,
        "stage": None,
        "stage_index": 0,
        "models": stage_models(quality),
        "events": [],
        "error": None,
        "result": None,
        "created": _now(),
        "updated": _now(),
    }
    with _lock:
        _gc_locked()
        _tasks[task_id] = record

    def on_event(event: dict) -> None:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return
            task["events"].append(event)
            task["updated"] = _now()
            if event.get("type") == "stage_start":
                task["stage"] = event.get("stage")
                task["stage_index"] = STAGES.index(event["stage"]) if event.get("stage") in STAGES else task["stage_index"]

    def runner() -> None:
        try:
            result = run_pipeline(audio_path, quality, title, author, on_event=on_event)
            with _lock:
                task = _tasks.get(task_id)
                if task:
                    task.update(status="done", stage=None, stage_index=len(STAGES),
                                result=result, updated=_now())
        except Exception as exc:  # noqa: BLE001
            with _lock:
                task = _tasks.get(task_id)
                if task:
                    task.update(status="error", error=f"{exc.__class__.__name__}: {exc}",
                                updated=_now())
        finally:
            audio_path.unlink(missing_ok=True)

    threading.Thread(target=runner, name=f"task-{task_id}", daemon=True).start()
    return task_id


def stats() -> dict:
    with _lock:
        running = sum(1 for t in _tasks.values() if t["status"] == "running")
        return {"tasks": len(_tasks), "running": running}
