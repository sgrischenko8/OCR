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

    save_crops: bool = True                  # зберігати кроп (уже після perspective-корекції) кожного розпізнаного регіону
    raw_crops_dir: Optional[Path] = None      # куди складати покадрові кропи; None -> "<log_path без .jsonl>_crops/"

    def __post_init__(self):
        # приводимо до Path на випадок, якщо передали звичайний рядок
        self.input_video = Path(self.input_video)
        self.output_video = Path(self.output_video)
        self.log_path = Path(self.log_path)

        if self.raw_crops_dir is not None:
            self.raw_crops_dir = Path(self.raw_crops_dir)
        elif self.save_crops:
            # напр. events.jsonl -> events_crops/, поруч із логом
            self.raw_crops_dir = self.log_path.parent / f"{self.log_path.stem}_crops"

        if not self.input_video.exists():
            raise FileNotFoundError(f"Вхідне відео не знайдено: {self.input_video}")

        if self.detector_type == "east":
            if not self.east_model_path:
                raise ValueError("Для detector_type='east' потрібно вказати east_model_path")
            self.east_model_path = Path(self.east_model_path)
            if not self.east_model_path.exists():
                raise FileNotFoundError(f"Модель EAST не знайдено: {self.east_model_path}")
