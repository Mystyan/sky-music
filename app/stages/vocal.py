from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import settings
from ..errors import ModelUnavailable
from ..mock import mock_transcription
from ..schemas import Note


def _midi_notes(path: Path) -> list[Note]:
    import mido

    midi = mido.MidiFile(str(path))
    notes: list[Note] = []
    for track in midi.tracks:
        t = 0.0
        active: dict[tuple[int, int], tuple[float, int]] = {}
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                key = (msg.channel, msg.note)
                if key in active:
                    start, velocity = active.pop(key)
                    notes.append(Note(pitch=msg.note, start=start, end=max(t, start + 0.05),
                                      velocity=velocity / 127.0, instrument="voice",
                                      source="midi"))
    return notes


def _chunks(path: Path, duration: float) -> list[tuple[float, float]]:
    """Split a vocal stem into 20-30s chunks with a 1s overlap."""
    try:
        import librosa

        y, sr = librosa.load(str(path), sr=16000, mono=True)
        intervals = librosa.effects.split(y, top_db=30)
        spans = [(a / sr, b / sr) for a, b in intervals]
    except Exception:
        spans = []

    if not spans:
        spans = [(0.0, duration)]

    chunks: list[tuple[float, float]] = []
    start = spans[0][0]
    end = spans[0][1]
    for a, b in spans[1:]:
        if b - start <= 28.0:
            end = b
        else:
            chunks.append((max(0.0, start - 1.0), min(duration, end + 1.0)))
            start, end = a, b
    chunks.append((max(0.0, start - 1.0), min(duration, end + 1.0)))
    return chunks or [(0.0, duration)]


def _vocalparse_notes(path: Path, duration: float, checkpoint: str) -> list[Note]:
    from vocalparse import transcribe_one  # type: ignore
    from vocalparse.evaluation import NOTE_DUR_MAP, SILENCE_WORDS  # type: ignore

    notes: list[Note] = []
    for start, end in _chunks(path, duration):
        clip = path.parent / f"vocal_{int(start * 1000)}_{int(end * 1000)}.wav"
        from ..audio import FFMPEG
        subprocess.run(
            [FFMPEG, "-y", "-ss", str(start), "-to", str(end), "-i", str(path),
             "-ar", "16000", "-ac", "1", str(clip)],
            capture_output=True, check=False,
        )
        parsed = transcribe_one(
            audio=str(clip), checkpoint=checkpoint, parse=True
        )
        if not parsed:
            continue
        bpm = float(parsed.get("bpm", 120)) or 120.0
        beat = 60.0 / bpm
        cursor = start
        for word, pitch, token in zip(parsed.get("words", []),
                                      parsed.get("pitches", []),
                                      parsed.get("notes", [])):
            beats = NOTE_DUR_MAP.get(token, 1.0)
            dur = beats * beat
            if word not in SILENCE_WORDS:
                notes.append(Note(pitch=int(pitch), start=cursor, end=cursor + dur,
                                  velocity=0.85, instrument="voice", source="vocalparse"))
            cursor += dur
    return _dedupe(notes)


def _game_notes(path: Path, duration: float, size: str = "large") -> list[Note]:
    game_dir = Path(settings.game_dir)
    model = game_dir / "models" / f"game-{size}.pt"
    if not model.exists():
        raise FileNotFoundError(f"GAME model not found: {model}")
    out = path.parent / "game_midi"
    out.mkdir(exist_ok=True)
    subprocess.run(
        ["python", "infer.py", "extract", str(path), "-m", str(model),
         "--output-formats", "mid"],
        cwd=str(game_dir), capture_output=True, check=True,
    )
    midi_files = list(path.parent.glob("*.mid")) + list(out.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError("GAME produced no MIDI")
    return _midi_notes(sorted(midi_files, key=lambda p: p.stat().st_mtime)[-1])


def _dedupe(notes: list[Note]) -> list[Note]:
    result: list[Note] = []
    for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if result and abs(result[-1].start - note.start) < 0.03 and abs(result[-1].pitch - note.pitch) <= 1:
            continue
        result.append(note)
    return result


def transcribe_vocal(
    path: str | Path, duration: float, spec: dict
) -> tuple[list[Note], str, list[str]]:
    path = Path(path)
    warnings: list[str] = []
    model = spec.get("vocal_model", "none")
    size = spec.get("vocal_size", "large")

    if settings.mock_mode:
        return mock_transcription(duration).melody, "mock", ["MOCK_MODE：人声使用内置样例旋律。"]

    if model == "basic_pitch":
        try:
            from .basicpitch import melody
            return melody(path), "basic_pitch", warnings
        except Exception as exc:
            raise ModelUnavailable(
                f"Basic Pitch 不可用：{exc.__class__.__name__}: {exc}"
            ) from exc

    if model == "vocalparse":
        try:
            from .. import models as model_store

            checkpoint = settings.vocalparse_ckpt or str(model_store.vocalparse_dir())
            return _vocalparse_notes(path, duration, checkpoint), "vocalparse", warnings
        except Exception as exc:
            raise ModelUnavailable(f"VocalParse 不可用：{exc.__class__.__name__}: {exc}") from exc

    if model in ("game", "auto"):
        if not settings.game_dir:
            raise ModelUnavailable("中/低音质档需要 GAME，但 GAME_DIR 未配置。")
        try:
            return _game_notes(path, duration, size), f"game-{size}", warnings
        except Exception as exc:
            raise ModelUnavailable(f"GAME 不可用：{exc.__class__.__name__}: {exc}") from exc

    raise ModelUnavailable(f"未知人声模型：{model}")
