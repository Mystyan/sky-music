"""Sky: Children of the Light 15-key helpers.

The in-game instruments use a 3x5 diatonic keyboard covering C4..C6.
SkyStudio names the 15 keys A1..A5, B1..B5, C1..C5 (index 0..14).
"""

from __future__ import annotations

from typing import Iterable

SKY_NAMES = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5",
]

# MIDI notes for the 15 keys, low to high: C4..C6 diatonic.
SKY_PITCHES = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84]

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def fold_to_sky(pitch: int, low: int = 60, high: int = 84) -> int:
    """Fold a MIDI pitch into the Sky range by octaves."""
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return pitch


def nearest_sky_index(pitch: int) -> int:
    """Return the closest of the 15 Sky keys, preferring the lower one on ties."""
    target = fold_to_sky(pitch)
    return min(range(len(SKY_PITCHES)), key=lambda i: (abs(SKY_PITCHES[i] - target), SKY_PITCHES[i]))


def best_key_shift(pitches: Iterable[int], max_shift: int = 6) -> int:
    """Choose a transposition that puts the most weighted notes on white keys."""
    counts: dict[int, float] = {}
    for p in pitches:
        counts[p % 12] = counts.get(p % 12, 0.0) + 1.0
    if not counts:
        return 0

    best_shift = 0
    best_score = -1.0
    for shift in range(-max_shift, max_shift + 1):
        score = sum(w for pc, w in counts.items() if (pc + shift) % 12 in MAJOR_SCALE)
        # Prefer smaller transpositions when the score is tied.
        if score > best_score or (score == best_score and abs(shift) < abs(best_shift)):
            best_score = score
            best_shift = shift
    return best_shift


def is_white_key(pitch: int, shift: int = 0) -> bool:
    return (pitch + shift) % 12 in MAJOR_SCALE


def pitch_class_name(pitch: int) -> str:
    return PITCH_CLASS_NAMES[pitch % 12]


def key_name(key: str, shift: int) -> str:
    """Transpose a key name such as C / Am by a number of semitones."""
    major = key.endswith("m") is False
    tonic = key[:-1] if key.endswith("m") else key
    if tonic not in PITCH_CLASS_NAMES:
        return key
    idx = (PITCH_CLASS_NAMES.index(tonic) + shift) % 12
    return PITCH_CLASS_NAMES[idx] + ("" if major else "m")
