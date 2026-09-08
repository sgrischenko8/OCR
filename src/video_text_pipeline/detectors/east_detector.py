from typing import List, Tuple

import cv2
import numpy as np

from .base import BaseTextDetector, TextRegion

class EASTTextDetector(BaseTextDetector):
    """
    Детектор на основі EAST (frozen_east_text_detection.pb).
    Вагу моделі в репозиторій не кладемо — завантажте її окремо і передайте
    шлях через config.east_model_path.
    """

    LAYER_NAMES = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (320, 320),
        min_confidence: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        self.net = cv2.dnn.readNet(model_path)
        self.input_size = input_size
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold

    def detect(self, frame: np.ndarray) -> List[TextRegion]:
        h, w = frame.shape[:2]
        new_w, new_h = self.input_size
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        scores, geometry = self.net.forward(self.LAYER_NAMES)

        boxes, confidences = self._decode(scores, geometry)
        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.min_confidence, self.nms_threshold)
        rx, ry = w / new_w, h / new_h

        results: List[TextRegion] = []
        for i in (indices.flatten() if len(indices) else []):
            (cx, cy), (bw, bh), angle = boxes[i]
            rect_pts = cv2.boxPoints(((cx * rx, cy * ry), (bw * rx, bh * ry), angle))
            results.append(TextRegion(polygon=rect_pts.astype(np.float32), confidence=confidences[i]))

        return results

    def _decode(self, scores, geometry):
        """Розбирає вихідні тензори EAST (RBOX формат) у обертові прямокутники."""
        num_rows, num_cols = scores.shape[2:4]
        boxes, confidences = [], []

        for y in range(num_rows):
            scores_row = scores[0, 0, y]
            x0, x1, x2, x3 = (geometry[0, i, y] for i in range(4))
            angles_row = geometry[0, 4, y]

            for x in range(num_cols):
                if scores_row[x] < self.min_confidence:
                    continue

                offset_x, offset_y = x * 4.0, y * 4.0
                angle = angles_row[x]
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                box_h = x0[x] + x2[x]
                box_w = x1[x] + x3[x]

                end_x = offset_x + cos_a * x1[x] + sin_a * x2[x]
                end_y = offset_y - sin_a * x1[x] + cos_a * x2[x]
                start_x, start_y = end_x - box_w, end_y - box_h
                center = ((start_x + end_x) / 2.0, (start_y + end_y) / 2.0)

                boxes.append((center, (box_w, box_h), -angle * 180.0 / np.pi))
                confidences.append(float(scores_row[x]))

        return boxes, confidences
