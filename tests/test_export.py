from app.schemas import Arrangement, Quality, SkyNote
from app.stages.export import to_abc15, to_skystudio


def _arrangement() -> Arrangement:
    return Arrangement(
        notes=[
            SkyNote(0, 0, 60, 1, "melody"),
            SkyNote(500, 7, 72, 1, "harmony"),
        ],
        bpm=120,
        key="C",
        shift=0,
        quality=Quality.medium,
        stats={},
    )


def test_skystudio_shape():
    data = to_skystudio(_arrangement(), "测试", "tester")[0]
    assert data["name"] == "测试"
    assert data["bpm"] == 120
    assert data["pitchLevel"] == 0
    assert data["bitsPerPage"] == 16
    assert data["isEncrypted"] is False
    assert data["songNotes"] == [
        {"time": 0, "key": "1Key0"},
        {"time": 500, "key": "1Key7"},
    ]
    assert data["notes"] == [[0, 0, 1], [7, 500, 1]]


def test_abc15_header_and_cells():
    text = to_abc15(_arrangement(), "测试", "tester")
    assert text.startswith("<DontCopyThisLine> 120 0 16")
    assert "A1" in text
    assert "B3" in text
