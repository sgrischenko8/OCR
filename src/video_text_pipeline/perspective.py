from typing import Optional

import cv2
import numpy as np

def order_points(pts: np.ndarray) -> np.ndarray:
    """Впорядковує 4 точки полігону як TL, TR, BR, BL — потрібно для стабільної warpPerspective."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def warp_text_region(frame: np.ndarray, polygon: np.ndarray, out_height: int = 64) -> Optional[np.ndarray]:
    """
    Вирівнює перспективу текстового регіону в прямокутник фіксованої висоти.
    Ширина рахується пропорційно до реальних розмірів полігону, щоб не
    спотворювати пропорції символів перед OCR.
    """
    rect = order_points(polygon)
    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    max_width = max(int(width_top), int(width_bottom))
    max_height = max(int(height_left), int(height_right))
    if max_width < 4 or max_height < 4:
        return None  # регіон занадто малий, найімовірніше — шум детектора

    out_width = max(int(max_width * (out_height / max_height)), 4)
    dst = np.array(
        [[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, matrix, (out_width, out_height))
