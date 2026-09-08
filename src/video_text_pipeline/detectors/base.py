from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class TextRegion:
    """Один знайдений текстовий регіон: полігон з 4 точок + впевненість детектора."""

    polygon: np.ndarray  # shape (4, 2), порядок точок довільний — впорядковується у perspective.py
    confidence: float

class BaseTextDetector(ABC):
    """Спільний інтерфейс для всіх детекторів тексту, щоб пайплайн не залежав від реалізації."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[TextRegion]:
        """Повертає список текстових регіонів, знайдених на кадрі."""
        raise NotImplementedError
