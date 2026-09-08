import logging
from pathlib import Path
from typing import Tuple

import cv2

logger = logging.getLogger(__name__)

class VideoReader:
    """Тонка обгортка над cv2.VideoCapture з базовою валідацією та метаданими відео."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Не вдалося відкрити відео: {self.path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

class VideoWriter:
    """
    Обгортка над cv2.VideoWriter. release() ідемпотентний і викликається
    завжди через finally у пайплайні — інакше при перериванні (Ctrl+C)
    контейнер mp4 не отримає moov-атом і файл вийде битим.
    """

    def __init__(self, path: Path, fps: float, size: Tuple[int, int], codec: str = "mp4v"):
        self.path = Path(path)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Не вдалося створити VideoWriter для {self.path}")
        self._closed = False

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        if not self._closed:
            self.writer.release()
            self._closed = True
            logger.info("Відеофайл коректно закрито: %s", self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
