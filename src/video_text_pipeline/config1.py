from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PipelineConfig:

    input_video: Path
    output_video: Path
    log_path: Path = Path("events.jsonl")

    detector_type: str = "easyocr"            # "easyocr" (дефолт, CRAFT), "mser" (без ваг) або "east"
    east_model_path: Optional[Path] = None    # frozen_east_text_detection.pb, якщо detector_type="east"

    min_confidence: float = 0.5  # поріг ocr_confidence; для mser/east ще й фільтрує полігони детектора
    ocr_lang: str = "en"      # мови для easyocr через кому, напр. "en,ru"
    use_gpu: bool = False     # прискорення на CUDA, якщо доступна відеокарта

    progress_log_step: int = 5               # логувати прогрес кожні N відсотків
    draw_annotations: bool = True            # малювати рамки знайденого тексту на вихідному відео
    max_text_regions_per_frame: int = 20     # запобіжник від зависання на шумних/складних кадрах
