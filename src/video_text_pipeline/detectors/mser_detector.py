from typing import List

import cv2
import numpy as np

from .base import BaseTextDetector, TextRegion

class MSERTextDetector(BaseTextDetector):
    """
    Легкий детектор тексту на MSER-регіонах. Точність нижча за EAST/CRAFT,
    зате працює одразу без завантаження ваг — зручно як дефолтний бекенд
    або для швидкого дебагу пайплайна на новому відео.
    """

    def __init__(self, min_area: int = 200, max_area: int = 15000, min_confidence: float = 0.3):
        self._mser = cv2.MSER_create()
        self._mser.setMinArea(min_area)
        self._mser.setMaxArea(max_area)
        self.min_confidence = min_confidence

    def detect(self, frame: np.ndarray) -> List[TextRegion]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        regions, _ = self._mser.detectRegions(gray)

        results: List[TextRegion] = []
        for pts in regions:
            x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))

            # відсіюємо явно не текстові пропорції (занадто квадратні або занадто витягнуті)
            aspect = w / max(h, 1)
            if not (0.1 < aspect < 15):
                continue

            polygon = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
            )

            # псевдо-впевненість: наскільки щільно регіон заповнює свій bounding box
            density = len(pts) / max(w * h, 1)
            confidence = float(np.clip(density * 2.0, 0.0, 1.0))
            if confidence < self.min_confidence:
                continue

            results.append(TextRegion(polygon=polygon, confidence=confidence))

        return results
