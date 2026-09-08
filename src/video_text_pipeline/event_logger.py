import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TextEvent:
    frame_index: int          # номер кадру у відео
    video_time_sec: float     # час у відео (сек), а не час обробки
    text: str                 # розпізнана фраза
    detector_confidence: float  # впевненість детектора тексту (полігон)
    ocr_confidence: float       # впевненість OCR по розпізнаному тексту
    bbox: List[List[float]]     # 4 кути полігону на кадрі, [[x, y], ...]
    bbox_center: List[float]    # центр полігону — зручно для фільтрації без парсингу bbox
    crop_path: Optional[str] = None  # шлях до збереженого кропу регіону (вже після perspective-корекції), якщо увімкнено

class EventLogger:
    """
    Пише події у JSONL (один JSON-об'єкт на рядок) — стрімінговий запис
    без буферизації всього лога в пам'яті, зручно парсити далі pandas/jq.

    Якщо переданий crops_dir, для кожної події додатково зберігається на
    диск кроп регіону (той самий, що йде в OCR — уже вирівняний
    perspective_correction), а шлях до нього пишеться в полі crop_path.
    Це покадрові "сирі" скріншоти; консолідатор далі сам вибирає, який саме
    кадр треку лишити як фінальний скріншот для events_consolidated.jsonl.
    """

    def __init__(self, path: Path, crops_dir: Optional[Path] = None):
        self.path = Path(path)
        self._file = open(self.path, "w", encoding="utf-8")
        self._count = 0

        self.crops_dir = Path(crops_dir) if crops_dir is not None else None
        if self.crops_dir is not None:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        frame_index: int,
        video_time_sec: float,
        text: str,
        detector_confidence: float,
        ocr_confidence: float,
        polygon: np.ndarray,
        crop: Optional[np.ndarray] = None,
        region_index: int = 0,
    ):
        center = polygon.mean(axis=0).tolist()
        crop_path = self._save_crop(frame_index, region_index, crop)
        event = TextEvent(
            frame_index=frame_index,
            video_time_sec=round(video_time_sec, 3),
            text=text,
            detector_confidence=round(float(detector_confidence), 3),
            ocr_confidence=round(float(ocr_confidence), 3),
            bbox=polygon.astype(float).round(1).tolist(),
            bbox_center=[round(c, 1) for c in center],
            crop_path=crop_path,
        )
        self._file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._count += 1

    def _save_crop(self, frame_index: int, region_index: int, crop: Optional[np.ndarray]) -> Optional[str]:
        """Зберігає кроп регіону на диск. Повертає шлях або None, якщо збереження вимкнене/не вдалося."""
        if self.crops_dir is None or crop is None:
            return None
        filename = f"f{frame_index:07d}_{region_index:02d}.png"
        crop_path = self.crops_dir / filename
        if not cv2.imwrite(str(crop_path), crop):
            logger.warning("Не вдалося зберегти кроп %s", crop_path)
            return None
        return str(crop_path)

    def close(self):
        self._file.flush()
        self._file.close()
        logger.info("Лог подій збережено: %s (%d записів)", self.path, self._count)
        if self.crops_dir is not None:
            logger.info("Покадрові кропи збережено в: %s", self.crops_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
