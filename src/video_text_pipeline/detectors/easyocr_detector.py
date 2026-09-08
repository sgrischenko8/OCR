from typing import List

import numpy as np

from .base import BaseTextDetector, TextRegion

class EasyOCRTextDetector(BaseTextDetector):
    """
    Детекція текстових регіонів через вбудовану CRAFT-модель easyocr
    (reader.detect()) — значно точніше за евристичний MSER, і той самий
    Reader далі повторно використовується в OCREngine.recognize().

    easyocr.detect() не повертає окрему впевненість на бокс, тому
    confidence тут завжди 1.0 — реальну якість показує ocr_confidence,
    який рахується вже після розпізнавання.
    """

    def __init__(self, reader, min_polygon_area: float = 50.0):
        self.reader = reader
        self.min_polygon_area = min_polygon_area

    def detect(self, frame: np.ndarray) -> List[TextRegion]:
        horizontal_list, free_list = self.reader.detect(frame)

        regions: List[TextRegion] = []

        for x_min, x_max, y_min, y_max in (horizontal_list[0] if horizontal_list else []):
            polygon = np.array(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                dtype=np.float32,
            )
            if self._area(polygon) >= self.min_polygon_area:
                regions.append(TextRegion(polygon=polygon, confidence=1.0))

        for points in (free_list[0] if free_list else []):
            polygon = np.array(points, dtype=np.float32)
            if self._area(polygon) >= self.min_polygon_area:
                regions.append(TextRegion(polygon=polygon, confidence=1.0))

        return regions

    @staticmethod
    def _area(polygon: np.ndarray) -> float:
        x = polygon[:, 0]
        y = polygon[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
