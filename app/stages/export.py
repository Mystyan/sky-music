from __future__ import annotations

import math

from ..schemas import Arrangement
from ..sky import SKY_NAMES


def to_skystudio(arrangement: Arrangement, title: str, author: str) -> list[dict]:
    song_notes = [
        {"time": int(n.time_ms), "key": f"{n.track}Key{n.key}"}
        for n in arrangement.notes
    ]
    notes = [[int(n.key), int(n.time_ms), int(n.track)] for n in arrangement.notes]
    return [{
        "name": title,
        "type": "composed",
        "folderId": None,
        "instruments": [{
            "name": "Harp",
            "volume": 100,
            "pitch": "",
            "visible": True,
            "icon": "circle",
            "alias": "",
            "muted": False,
            "reverbOverride": None,
        }],
        "version": 2,
        "pitch": arrangement.key,
        "bpm": int(round(arrangement.bpm)),
        "reverb": False,
        "data": {"isComposed": True, "isComposedVersion": True, "appName": "Sky"},
        "notes": notes,
        "id": None,
        "isComposed": True,
        "pitchLevel": 0,
        "bitsPerPage": 16,
        "isEncrypted": False,
        "songNotes": song_notes,
    }]


def to_abc15(arrangement: Arrangement, title: str, author: str) -> str:
    bpm = max(1, int(round(arrangement.bpm)))
    tick_ms = 4 * 60000 / bpm / 16
    max_tick = max((int(math.ceil(n.time_ms / tick_ms)) for n in arrangement.notes), default=0) + 1
    cells: list[set[str]] = [set() for _ in range(max_tick)]
    for note in arrangement.notes:
        tick = min(int(round(note.time_ms / tick_ms)), max_tick - 1)
        cells[tick].add(SKY_NAMES[note.key])

    header = f"<DontCopyThisLine> {bpm} 0 16 {author or 'sky-music'} {author or 'sky-music'} {title}\n"
    lines = []
    for i in range(0, len(cells), 16):
        chunk = cells[i:i + 16]
        lines.append(" ".join("".join(sorted(c)) if c else "." for c in chunk))
    return header + "\n".join(lines) + "\n"
