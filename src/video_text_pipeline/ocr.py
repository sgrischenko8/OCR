from typing import Tuple

import numpy as np

class OCREngine:
    """
    Обгортка над easyocr.Reader.recognize() — навмисно НЕ readtext().

    readtext() ганяє повний конвеєр (власна CRAFT-детекція + розпізнавання),
    а crop, який сюди приходить, уже щільно вирівняний perspective_correction
    під один рядок тексту. Запускати ще одну детекцію поверх такого кропу
    сенсу нема: без полів навколо тексту детектор часто взагалі не знаходить
    жодного боксу і повертає порожній результат — звідси "нічого не
    розпізнає". recognize() пропускає детекцію і розпізнає весь переданий
    кроп як один рядок напряму.

    Reader приймається ззовні (спільний з детектором), щоб ваги моделі
    завантажувались і трималися в пам'яті один раз, а не двічі.
    """

    def __init__(self, reader):
        self.reader = reader

    def recognize(self, crop: np.ndarray) -> Tuple[str, float]:
        """Повертає (текст, впевненість 0..1). Якщо нічого не розпізнано — ("", 0.0)."""
        results = self.reader.recognize(crop)
        if not results:
            return "", 0.0

        _, text, confidence = results[0]
        text = text.strip()
        if not text:
            return "", 0.0

        return text, float(confidence)
