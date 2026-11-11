from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Quality(str, Enum):
    """Quality tier: controls model size/quality first, arrangement density second."""

    low = "low"
    medium = "medium"
    high = "high"


QUALITY_SPECS = {
    Quality.low: {
        "label": "低",
        "description": "轻量模型，速度优先：Demucs + Basic Pitch，单音输出。",
        "separator": "demucs",
        "vocal_model": "basic_pitch",
        "vocal_size": "small",
        "instrument_model": "basic_pitch",
        "instrument_size": "small",
        "quantize_div": 2,
        "max_notes_per_second": 4.0,
        "harmony_voices": 1,
        "harmony_onsets": "downbeat",
    },
    Quality.medium: {
        "label": "中",
        "description": "标准模型：SubRoformer + GAME-medium + MuScriptor-medium，双音输出。",
        "separator": "subroformer",
        "vocal_model": "game",
        "vocal_size": "medium",
        "instrument_model": "muscriptor",
        "instrument_size": "medium",
        "quantize_div": 4,
        "max_notes_per_second": 8.0,
        "harmony_voices": 2,
        "harmony_onsets": "beat",
    },
    Quality.high: {
        "label": "高",
        "description": "最高质量：SubRoformer + VocalParse-1.7B + GAME-large + MuScriptor-large，三音输出。",
        "separator": "subroformer",
        "vocal_model": "vocalparse",
        "vocal_size": "large",
        "instrument_model": "muscriptor",
        "instrument_size": "large",
        "quantize_div": 4,
        "max_notes_per_second": 12.0,
        "harmony_voices": 3,
        "harmony_onsets": "eighth",
    },
}


@dataclass
class Note:
    pitch: int
    start: float
    end: float
    velocity: float = 0.8
    instrument: str = "unknown"
    source: str = "unknown"

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass
class Track:
    name: str
    instrument: str
    notes: list[Note] = field(default_factory=list)


@dataclass
class Transcription:
    melody: list[Note] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chord:
    start: float
    end: float
    symbol: str
    pitch_classes: list[int]


@dataclass
class Analysis:
    bpm: float = 120.0
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    key: str = "C"
    chords: list[Chord] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0


@dataclass
class SkyNote:
    time_ms: int
    key: int                  # 0..14
    pitch: int
    track: int = 1
    source: str = "melody"


@dataclass
class Arrangement:
    notes: list[SkyNote]
    bpm: float
    key: str
    shift: int
    quality: Quality
    stats: dict[str, Any] = field(default_factory=dict)


class TranscribeUrlRequest(BaseModel):
    audio_url: str = Field(..., description="可直接下载的音频 URL")
    quality: Quality = Quality.medium
    title: str = "未命名"
    author: str = ""


class QualityInfo(BaseModel):
    id: str
    label: str
    description: str
    vocal_model: str
    instrument_model: str
