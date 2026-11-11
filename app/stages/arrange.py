from __future__ import annotations

from ..schemas import (
    Analysis,
    Arrangement,
    Chord,
    Note,
    Quality,
    QUALITY_SPECS,
    SkyNote,
    Track,
)
from ..sky import (
    SKY_PITCHES,
    best_key_shift,
    fold_to_sky,
    is_white_key,
    key_name,
    nearest_sky_index,
)


def _quantize(t: float, grid: float) -> float:
    return round(t / grid) * grid


def _melody_notes(melody: list[Note], shift: int, grid: float, spec: dict) -> list[SkyNote]:
    grouped: dict[float, Note] = {}
    for note in melody:
        if note.duration < 0.05:
            continue
        t = round(_quantize(note.start, grid), 4)
        current = grouped.get(t)
        if current is None or note.pitch > current.pitch:
            grouped[t] = note

    min_gap = 1.0 / max(spec["max_notes_per_second"], 1.0)
    result: list[SkyNote] = []
    for t in sorted(grouped):
        if result and t - result[-1].time_ms / 1000.0 < min_gap * 0.5:
            continue
        note = grouped[t]
        pitch = fold_to_sky(note.pitch + shift)
        result.append(SkyNote(time_ms=int(round(t * 1000)),
                              key=nearest_sky_index(pitch),
                              pitch=pitch, track=1, source="melody"))
    return result


def _chord_onsets(chord: Chord, beats: list[float], quality: Quality, beat: float) -> list[float]:
    onsets = {round(chord.start, 4)}
    for t in beats:
        if chord.start - 0.01 <= t < chord.end - 0.01:
            if quality == Quality.low:
                # Only the first beat of each bar.
                if abs(t - chord.start) < 0.02:
                    onsets.add(round(t, 4))
            elif quality == Quality.medium:
                if abs((t - chord.start) / beat - round((t - chord.start) / beat)) < 0.1:
                    if round((t - chord.start) / beat) % 2 == 0:
                        onsets.add(round(t, 4))
            else:
                onsets.add(round(t, 4))
    return sorted(onsets)


def _voicing(chord: Chord, melody_pitch: int | None, voices: int, previous: list[int]) -> list[int]:
    candidates: list[int] = []
    for pc in chord.pitch_classes:
        for octave in range(4, 7):
            pitch = 12 * octave + pc
            if 60 <= pitch <= 84:
                candidates.append(pitch)
    candidates = sorted(set(candidates))

    # Keep the harmony below the melody when possible.
    if melody_pitch is not None:
        below = [p for p in candidates if p <= melody_pitch - 2]
        if below:
            candidates = below

    # Prefer a compact voicing near the previous one.
    target = sum(previous) / len(previous) if previous else 64
    candidates.sort(key=lambda p: (abs(p - target), p))

    chosen: list[int] = []
    for pitch in candidates:
        if melody_pitch is not None and abs(pitch - melody_pitch) == 0:
            continue
        if all(abs(pitch - other) >= 2 for other in chosen):
            chosen.append(pitch)
        if len(chosen) >= voices:
            break
    return sorted(chosen)


def arrange(
    melody: list[Note],
    tracks: list[Track],
    analysis: Analysis,
    quality: Quality,
) -> Arrangement:
    spec = QUALITY_SPECS[quality]
    beat = 60.0 / analysis.bpm if analysis.bpm > 0 else 0.5
    grid = beat / spec["quantize_div"]

    all_pitches = [n.pitch for n in melody]
    for track in tracks:
        all_pitches.extend(n.pitch for n in track.notes)
    shift = best_key_shift(all_pitches)

    melody_notes = _melody_notes(melody, shift, grid, spec)
    melody_at = {n.time_ms: n.pitch for n in melody_notes}
    melody_times = sorted(melody_at)

    chords = analysis.chords or [Chord(0.0, max(analysis.duration, beat * 4), "C", [0, 4, 7])]
    output = list(melody_notes)
    previous: list[int] = []
    used_keys: dict[int, float] = {}

    def melody_near(t_ms: int) -> int | None:
        candidates = [m for m in melody_times if abs(m - t_ms) <= 35]
        return melody_at[candidates[0]] if candidates else None

    def can_add(t_ms: int, key: int) -> bool:
        if key in used_keys and t_ms / 1000.0 - used_keys[key] < 0.07:
            return False
        simultaneous = sum(1 for n in output if abs(n.time_ms - t_ms) <= 30)
        return simultaneous < spec["max_simultaneous"]

    for chord in chords:
        for onset in _chord_onsets(chord, analysis.beats, quality, beat):
            t = _quantize(onset, grid)
            if t < 0 or t > analysis.duration:
                continue
            t_ms = int(round(t * 1000))
            melody_pitch = melody_near(t_ms)
            voices = spec["harmony_voices"]
            pitches = _voicing(chord, melody_pitch, voices, previous)
            added: list[int] = []
            for pitch in pitches:
                key = nearest_sky_index(pitch)
                if not can_add(t_ms, key):
                    continue
                output.append(SkyNote(time_ms=t_ms, key=key, pitch=pitch,
                                      track=1, source="harmony"))
                used_keys[key] = t_ms / 1000.0
                added.append(pitch)
            if added:
                previous = added

    output.sort(key=lambda n: (n.time_ms, n.pitch))

    times = [n.time_ms for n in output]
    max_sim = 0
    for t in set(times):
        max_sim = max(max_sim, sum(1 for x in times if abs(x - t) <= 30))

    chromatic = sum(1 for n in melody if not is_white_key(n.pitch, shift))
    stats = {
        "notes": len(output),
        "melody_notes": len(melody_notes),
        "harmony_notes": len(output) - len(melody_notes),
        "max_simultaneous": max_sim,
        "chromatic_input_notes": chromatic,
        "shift": shift,
        "range": [min(SKY_PITCHES[n.key] for n in output), max(SKY_PITCHES[n.key] for n in output)] if output else [],
        "quality": quality.value,
    }

    return Arrangement(
        notes=output,
        bpm=analysis.bpm,
        key=key_name(analysis.key, shift),
        shift=shift,
        quality=quality,
        stats=stats,
    )
