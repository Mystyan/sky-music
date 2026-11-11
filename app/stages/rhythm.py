from __future__ import annotations

import numpy as np

from ..schemas import Analysis


def analyze(y: np.ndarray, sr: int, duration: float) -> Analysis:
    """Beat/BPM analysis with librosa; falls back to a fixed 120 BPM grid."""
    try:
        import librosa

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
        bpm = float(np.atleast_1d(tempo)[0])
        beats = [float(t) for t in np.atleast_1d(beat_frames)]
    except Exception:
        bpm, beats = 120.0, []

    if not beats:
        bpm = bpm if bpm > 0 else 120.0
        beat = 60.0 / bpm
        beats = [i * beat for i in range(int(duration / beat) + 2)]

    if bpm <= 0:
        bpm = 120.0

    downbeats = beats[::4]
    bar = 60.0 / bpm * 4
    sections = []
    for i in range(int(duration // (bar * 16)) + 1):
        start = i * bar * 16
        sections.append({"start": start, "end": min(duration, start + bar * 16),
                         "label": "intro" if i == 0 else ("chorus" if i % 2 == 0 else "verse")})

    return Analysis(bpm=bpm, beats=beats, downbeats=downbeats, duration=duration, sections=sections)
