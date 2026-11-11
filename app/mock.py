from __future__ import annotations

from .schemas import Note, Track, Transcription

BPM = 100.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT

# C major: C Am F G, one bar each.
PROGRESSION = [
    ("C",  [60, 64, 67], 48),
    ("Am", [57, 60, 64], 45),
    ("F",  [53, 57, 60], 41),
    ("G",  [55, 59, 62], 43),
]

MELODY = [72, 74, 76, 79, 76, 74, 72, 71, 72, 74, 76, 74, 72, 71, 69, 67]


def mock_transcription(duration: float) -> Transcription:
    """A deterministic C-major sketch used when real models are unavailable."""
    duration = max(duration, BAR)
    melody: list[Note] = []
    chords: list[Note] = []
    bass: list[Note] = []

    bars = int(duration // BAR) + 1
    for bar in range(bars):
        bar_start = bar * BAR
        name, triad, bass_note = PROGRESSION[bar % len(PROGRESSION)]

        # Two melody notes per beat, one octave around C5.
        for step in range(8):
            start = bar_start + step * (BEAT / 2)
            if start >= duration:
                break
            pitch = MELODY[(bar * 8 + step) % len(MELODY)]
            melody.append(Note(pitch=pitch, start=start, end=start + BEAT / 2 * 0.9,
                               velocity=0.85, instrument="voice", source="mock"))

        # Chords on beats 1 and 3.
        for beat in (0, 2):
            start = bar_start + beat * BEAT
            if start >= duration:
                break
            for pitch in triad:
                chords.append(Note(pitch=pitch, start=start, end=start + 2 * BEAT,
                                   velocity=0.55, instrument="piano", source="mock"))

        # Bass root on the downbeat.
        bass.append(Note(pitch=bass_note, start=bar_start, end=bar_start + BAR,
                         velocity=0.6, instrument="bass", source="mock"))

    return Transcription(
        melody=melody,
        tracks=[
            Track(name="piano", instrument="piano", notes=chords),
            Track(name="bass", instrument="bass", notes=bass),
        ],
        warnings=["MOCK_MODE=1：当前输出为内置 C 大调样例，不是真实扒谱结果。"],
    )
