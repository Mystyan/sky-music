from __future__ import annotations

import json
import os
from pathlib import Path

import mido

from ..schemas import Analysis, Arrangement, Note, Quality, SkyNote, Track
from ..sky import SKY_PITCHES
from vendor.sky_scores import arrange as repo

TICKS = 480


def _write_midi(path: Path, melody: list[Note], tracks: list[Track], bpm: float) -> None:
    bass = [n for t in tracks if "bass" in t.instrument.lower() for n in t.notes]
    harm = [n for t in tracks if "bass" not in t.instrument.lower() for n in t.notes]
    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS)
    for idx, notes in enumerate([melody, [], bass, harm]):
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        if idx == 0:
            tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm or 120), time=0))
        events = []
        for note in notes:
            on = int(note.start * (bpm or 120) / 60 * TICKS)
            off = max(int(note.end * (bpm or 120) / 60 * TICKS), on + 1)
            events.append((on, 1, note.pitch, note.velocity))
            events.append((off, 0, note.pitch, 0.0))
        events.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for tick, is_on, pitch, velocity in events:
            delta = max(tick - last, 0)
            last = tick
            tr.append(mido.Message(
                "note_on" if is_on else "note_off",
                note=int(pitch),
                velocity=max(1, min(127, int(velocity * 127))) if is_on else 0,
                time=delta,
            ))
        tr.append(mido.MetaMessage("end_of_track", time=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))


def arrange_from_midi(
    melody: list[Note],
    tracks: list[Track],
    analysis: Analysis,
    quality: Quality,
    title: str = "sky-music",
    author: str = "",
) -> Arrangement:
    """Use the vendored sky-scores arranger, exactly like its MIDI path."""
    workdir = Path(os.environ.get("TEMP_DIR", "/tmp/sky-music")) / "repo_arrange"
    workdir.mkdir(parents=True, exist_ok=True)
    name = "input"
    midi_path = workdir / "mid" / f"{name}.mid"
    _write_midi(midi_path, melody, tracks, analysis.bpm)

    bar = 60.0 / (analysis.bpm or 120) * 4

    def level(m: int) -> int:
        t = m * bar
        for section in analysis.sections:
            if section["start"] <= t < section["end"]:
                label = section.get("label", "")
                return 0 if label == "intro" else 2 if label == "chorus" else 1
        return 1

    ref = Path(__file__).resolve().parents[2] / "vendor" / "sky_scores" / "template.txt"
    out = workdir / "out.txt"
    old = Path.cwd()
    try:
        os.chdir(workdir)
        repo.arrange(
            f"mid:{name}", vocal=0, leads=[], bass=2,
            name=title, dst=str(out), ref=str(ref), harm=[3], level=level,
        )
    finally:
        os.chdir(old)

    data = json.loads(out.read_text(encoding="utf-16"))[0]
    data["name"] = title
    data["author"] = author
    notes: list[SkyNote] = []
    for item in data.get("songNotes", []):
        key = int(str(item["key"]).split("Key")[-1])
        notes.append(SkyNote(time_ms=int(item["time"]), key=key,
                             pitch=SKY_PITCHES[key], track=1, source="sky-scores"))

    times = [n.time_ms for n in notes]
    max_sim = 0
    for t in set(times):
        max_sim = max(max_sim, sum(1 for x in times if abs(x - t) <= 30))

    return Arrangement(
        notes=notes,
        bpm=analysis.bpm,
        key=analysis.key,
        shift=0,
        quality=quality,
        stats={
            "notes": len(notes),
            "max_simultaneous": max_sim,
            "quality": quality.value,
            "arranger": "sky-scores",
        },
    )
