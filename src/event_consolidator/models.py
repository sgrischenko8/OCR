from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

@dataclass
class RawEvent:
    """Один запис із покадрового event-логера (як пише event_logger.py в основному пайплайні)."""

    frame_index: int
    video_time_sec: float
    text: str
    detector_confidence: float
    ocr_confidence: float
    bbox: np.ndarray  # (4, 2) — кути полігону на кадрі
    crop_path: Optional[str] = None  # шлях до покадрового "сирого" скріншота регіону, якщо збережений

@dataclass
class Track:
    """Послідовність спостережень однієї й тієї ж фрази на кількох кадрах поспіль."""

    observations: List[RawEvent] = field(default_factory=list)

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def last_text(self) -> str:
        return self.observations[-1].text

    @property
    def last_bbox(self) -> np.ndarray:
        return self.observations[-1].bbox
