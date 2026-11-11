from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..schemas import Analysis, Chord, Note, Track
from ..sky import PITCH_CLASS_NAMES

KRUMHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]


def detect_key(y: np.ndarray, sr: int) -> str:
    try:
        import librosa

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        profile = np.mean(chroma, axis=1)
        profile = profile / (np.linalg.norm(profile) + 1e-9)

        best = (-1.0, "C")
        for tonic in range(12):
            for mode, template, suffix in (
                ("major", KRUMHANSL_MAJOR, ""),
                ("minor", KRUMHANSL_MINOR, "m"),
            ):
                rotated = np.roll(np.array(template, dtype=float), tonic)
                rotated = rotated / (np.linalg.norm(rotated) + 1e-9)
                score = float(np.dot(profile, rotated))
                if score > best[0]:
                    best = (score, PITCH_CLASS_NAMES[tonic] + suffix)
        return best[1]
    except Exception:
        return "C"


def _scale_for_key(key: str) -> tuple[int, list[int], bool]:
    minor = key.endswith("m")
    tonic = key[:-1] if minor else key
    pc = PITCH_CLASS_NAMES.index(tonic) if tonic in PITCH_CLASS_NAMES else 0
    return pc, (MINOR if minor else MAJOR), minor


def _chord_templates(key: str) -> list[tuple[str, list[int]]]:
    tonic, scale, minor = _scale_for_key(key)
    templates = []
    for degree in range(7):
        root = (tonic + scale[degree]) % 12
        third = (tonic + scale[(degree + 2) % 7]) % 12
        fifth = (tonic + scale[(degree + 4) % 7]) % 12
        if (third - root) % 12 == 3:
            quality = "m"
        elif (third - root) % 12 == 3 and (fifth - root) % 12 == 6:
            quality = "dim"
        elif (third - root) % 12 == 6:
            quality = "dim"
        else:
            quality = ""
        if minor and degree == 0:
            quality = "m"
        templates.append((PITCH_CLASS_NAMES[root] + quality, [root, third, fifth]))
    return templates


def infer_chords(tracks: list[Track], analysis: Analysis, duration: float) -> list[Chord]:
    """Infer one diatonic chord per bar from the transcribed notes."""
    templates = _chord_templates(analysis.key)
    bar = 60.0 / analysis.bpm * 4 if analysis.bpm > 0 else 2.0
    events: list[tuple[float, float, Note, float]] = []
    for track in tracks:
        weight = 1.6 if track.instrument in ("bass", "electric_bass", "acoustic_bass") else 1.0
        for note in track.notes:
            events.append((note.start, note.end, note, weight))

    chords: list[Chord] = []
    previous_pcs: list[int] | None = None
    start = 0.0
    while start < max(duration, bar):
        end = min(start + bar, duration) if duration else start + bar
        weights: dict[int, float] = defaultdict(float)
        for n_start, n_end, note, weight in events:
            overlap = max(0.0, min(end, n_end) - max(start, n_start))
            if overlap > 0:
                weights[note.pitch % 12] += overlap * weight

        best = None
        for symbol, pcs in templates:
            coverage = sum(weights.get(pc, 0.0) for pc in pcs)
            total = sum(weights.values()) + 1e-9
            root_bonus = 0.15 if weights.get(pcs[0], 0.0) > 0 else 0.0
            smooth = 0.05 if previous_pcs and set(pcs) == set(previous_pcs) else 0.0
            score = coverage / total + root_bonus + smooth
            if best is None or score > best[0]:
                best = (score, symbol, pcs)

        if best is None:
            best = (0.0, "C", [0, 4, 7])
        chords.append(Chord(start=start, end=end, symbol=best[1], pitch_classes=best[2]))
        previous_pcs = best[2]
        start = end
        if start >= max(duration, bar):
            break

    return chords
