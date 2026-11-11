from __future__ import annotations

from pathlib import Path

from ..schemas import Note, Track


def _events(path: str | Path):
    from basic_pitch.inference import predict  # type: ignore

    _, _, note_events = predict(str(path))
    return note_events


def melody(path: str | Path) -> list[Note]:
    """Lightweight monophonic melody: highest note per onset cluster."""
    notes: list[Note] = []
    for start, end, pitch, amplitude, _bend in _events(path):
        if amplitude < 0.15:
            continue
        notes.append(Note(pitch=int(round(pitch)), start=float(start), end=float(end),
                          velocity=float(amplitude), instrument="voice",
                          source="basic_pitch"))

    notes.sort(key=lambda n: (n.start, -n.pitch))
    result: list[Note] = []
    for note in notes:
        if result and abs(result[-1].start - note.start) < 0.03:
            if note.pitch > result[-1].pitch:
                result[-1] = note
        else:
            result.append(note)
    return result


def tracks(path: str | Path) -> list[Track]:
    """Lightweight instrument track; Basic Pitch is instrument-agnostic."""
    notes = [
        Note(pitch=int(round(pitch)), start=float(start), end=float(end),
             velocity=float(amplitude), instrument="piano", source="basic_pitch")
        for start, end, pitch, amplitude, _bend in _events(path)
        if amplitude >= 0.15
    ]
    return [Track(name="basic_pitch", instrument="piano", notes=notes)] if notes else []
