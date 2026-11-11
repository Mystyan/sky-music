from app.schemas import Analysis, Chord, Note, Quality, QUALITY_SPECS, Track
from app.stages.arrange import arrange


def _analysis() -> Analysis:
    beats = [i * 0.6 for i in range(17)]
    return Analysis(
        bpm=100,
        beats=beats,
        downbeats=beats[::4],
        key="C",
        duration=9.6,
        chords=[
            Chord(0.0, 2.4, "C", [0, 4, 7]),
            Chord(2.4, 4.8, "Am", [9, 0, 4]),
            Chord(4.8, 7.2, "F", [5, 9, 0]),
            Chord(7.2, 9.6, "G", [7, 11, 2]),
        ],
    )


def _melody() -> list[Note]:
    return [
        Note(72, i * 0.3, i * 0.3 + 0.25, instrument="voice", source="test")
        for i in range(30)
    ]


def _tracks() -> list[Track]:
    notes = [
        Note(60, i * 0.6, i * 0.6 + 0.5, instrument="piano", source="test")
        for i in range(16)
    ]
    bass = [Note(48, i * 2.4, i * 2.4 + 2.0, instrument="bass", source="test") for i in range(4)]
    return [Track("piano", "piano", notes), Track("bass", "bass", bass)]


def test_all_qualities_respect_simultaneous_limit():
    for quality in Quality:
        result = arrange(_melody(), _tracks(), _analysis(), quality)
        assert result.notes
        assert result.stats["max_simultaneous"] <= QUALITY_SPECS[quality]["max_simultaneous"]
        assert result.stats["range"][0] >= 60
        assert result.stats["range"][1] <= 84
        assert all(0 <= n.key <= 14 for n in result.notes)


def test_notes_sorted_and_unique():
    result = arrange(_melody(), _tracks(), _analysis(), Quality.medium)
    times = [(n.time_ms, n.key) for n in result.notes]
    assert times == sorted(times)
    assert len(times) == len(set(times))
