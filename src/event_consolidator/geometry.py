from typing import List, Tuple

import numpy as np

def bounding_rect(polygon: np.ndarray) -> Tuple[float, float, float, float]:
    """Описаний прямокутник полігону: (x_min, y_min, x_max, y_max)."""
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    return float(x_min), float(y_min), float(x_max), float(y_max)

def iou(polygon_a: np.ndarray, polygon_b: np.ndarray) -> float:
    """
    IoU по описаних прямокутниках, а не по точних полігонах — для трекінгу
    тексту цього достатньо і значно стабільніше при невеликому "тремтінні"
    кутів полігону між кадрами.
    """
    ax1, ay1, ax2, ay2 = bounding_rect(polygon_a)
    bx1, by1, bx2, by2 = bounding_rect(polygon_b)

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0

def average_polygon(polygons: List[np.ndarray]) -> np.ndarray:
    """Поелементне середнє кількох полігонів (4 точки) — усереднена позиція треку на екрані."""
    return np.stack(polygons, axis=0).mean(axis=0)
