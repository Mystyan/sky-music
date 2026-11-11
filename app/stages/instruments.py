from __future__ import annotations

from pathlib import Path

import mido

from ..config import settings
from ..errors import ModelUnavailable
from ..mock import mock_transcription
from ..schemas import Note, Track

_GM_FAMILY = {
    range(0, 8): "piano",
    range(8, 16): "chromatic",
    range(16, 24): "organ",
    range(24, 32): "guitar",
    range(32, 40): "bass",
    range(40, 48): "strings",
    range(48, 56): "strings",
    range(56, 64): "brass",
    range(64, 72): "woodwinds",
    range(72, 80): "woodwinds",
    range(80, 88): "synth",
}


def _family(program: int) -> str:
    for rng, name in _GM_FAMILY.items():
        if program in rng:
            return name
    return "other"


def _parse_midi(path: Path) -> list[Track]:
    midi = mido.MidiFile(str(path))
    tracks: list[Track] = []
    for idx, midi_track in enumerate(midi.tracks):
        program = 0
        t = 0.0
        active: dict[tuple[int, int], tuple[float, int]] = {}
        notes: list[Note] = []
        for msg in midi_track:
            t += msg.time
            if msg.type == "program_change":
                program = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                key = (msg.channel, msg.note)
                if key in active:
                    start, velocity = active.pop(key)
                    notes.append(Note(
                        pitch=msg.note,
                        start=start,
                        end=max(t, start + 0.05),
                        velocity=velocity / 127.0,
                        instrument=_family(program),
                        source="muscriptor",
                    ))
        if notes:
            tracks.append(Track(name=f"track{idx}_{_family(program)}",
                                instrument=_family(program), notes=notes))
    return tracks


def transcribe_instruments(
    path: str | Path, duration: float, spec: dict
) -> tuple[list[Track], str, list[str]]:
    path = Path(path)
    warnings: list[str] = []
    model_kind = spec.get("instrument_model", "none")
    size = spec.get("instrument_size", "large")

    if settings.mock_mode:
        return mock_transcription(duration).tracks, "mock", ["MOCK_MODE：伴奏使用内置样例和弦。"]

    if model_kind == "basic_pitch":
        try:
            from .basicpitch import tracks
            result = tracks(path)
            if not result:
                raise ModelUnavailable("Basic Pitch 输出为空。")
            return result, "basic_pitch", warnings
        except ModelUnavailable:
            raise
        except Exception as exc:
            raise ModelUnavailable(
                f"Basic Pitch 不可用：{exc.__class__.__name__}: {exc}"
            ) from exc

    if model_kind not in ("auto", "muscriptor"):
        raise ModelUnavailable(f"未知多乐器模型：{model_kind}")

    try:
        from muscriptor import TranscriptionModel  # type: ignore

        model = TranscriptionModel.load_model(size)
        midi_bytes = model.transcribe_to_midi(str(path))
        midi_path = path.parent / "muscriptor.mid"
        midi_path.write_bytes(midi_bytes)
        tracks = _parse_midi(midi_path)
        if not tracks:
            raise ModelUnavailable("MuScriptor 输出为空。")
        return tracks, f"muscriptor-{size}", warnings
    except ModelUnavailable:
        raise
    except Exception as exc:
        raise ModelUnavailable(
            f"MuScriptor-{size} 不可用：{exc.__class__.__name__}: {exc}"
        ) from exc
