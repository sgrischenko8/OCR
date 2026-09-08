import logging

import cv2
import easyocr
import numpy as np

from .config import PipelineConfig
from .detectors.base import BaseTextDetector
from .detectors.east_detector import EASTTextDetector
from .detectors.easyocr_detector import EasyOCRTextDetector
from .detectors.mser_detector import MSERTextDetector
from .event_logger import EventLogger
from .ocr import OCREngine
from .perspective import warp_text_region
from .video_io import VideoReader, VideoWriter

logger = logging.getLogger(__name__)

def build_ocr_reader(cfg: PipelineConfig) -> easyocr.Reader:
    languages = [code.strip() for code in cfg.ocr_lang.split(",") if code.strip()]
    return easyocr.Reader(languages, gpu=cfg.use_gpu, verbose=False)

def build_detector(cfg: PipelineConfig, ocr_reader: easyocr.Reader) -> BaseTextDetector:
    if cfg.detector_type == "east":
        if not cfg.east_model_path:
            raise ValueError("Для detector_type='east' потрібно вказати east_model_path")
        return EASTTextDetector(str(cfg.east_model_path), min_confidence=cfg.min_confidence)
    if cfg.detector_type == "mser":
        return MSERTextDetector(min_confidence=cfg.min_confidence)
    # "easyocr" (дефолт) — той самий Reader, що й для розпізнавання, тому ваги
    # моделі завантажуються й тримаються в пам'яті один раз, а не двічі
    return EasyOCRTextDetector(ocr_reader)

class VideoTextPipeline:
    """
    Video -> Text Detection -> Polygon/Conf. -> Perspective Correction -> OCR -> Event Logger.

    На екран нічого не виводиться: прогрес йде в лог, результат — у вихідне
    відео (за бажанням з розміткою) та JSONL з подіями.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        ocr_reader = build_ocr_reader(cfg)
        self.detector = build_detector(cfg, ocr_reader)
        self.ocr = OCREngine(ocr_reader)
        self._stop_requested = False

    def request_stop(self):
        """Викликається обробником сигналу — цикл обробки зупиниться на початку наступної ітерації."""
        self._stop_requested = True

    def run(self):
        reader = VideoReader(self.cfg.input_video)
        writer = VideoWriter(self.cfg.output_video, reader.fps, (reader.width, reader.height))
        crops_dir = self.cfg.raw_crops_dir if self.cfg.save_crops else None
        event_logger = EventLogger(self.cfg.log_path, crops_dir=crops_dir)

        total_frames = reader.frame_count or 1
        last_logged_percent = -1

        try:
            for frame_index, frame in enumerate(reader):
                if self._stop_requested:
                    logger.warning(
                        "Отримано сигнал зупинки, завершуємо коректно на кадрі %d/%d",
                        frame_index, total_frames,
                    )
                    break

                video_time = frame_index / reader.fps
                self._process_frame(frame, frame_index, video_time, event_logger)
                writer.write(frame)

                percent = int((frame_index + 1) / total_frames * 100)
                if percent >= last_logged_percent + self.cfg.progress_log_step:
                    logger.info("Оброблено %d%% (%d/%d кадрів)", percent, frame_index + 1, total_frames)
                    last_logged_percent = percent

        finally:
            # release() тут обов'язковий у будь-якому сценарії виходу (штатному, по
            # винятку чи по Ctrl+C) — інакше вихідний mp4 залишиться пошкодженим.
            reader.release()
            writer.release()
            event_logger.close()

    def _process_frame(
        self, frame: np.ndarray, frame_index: int, video_time: float, event_logger: EventLogger
    ):
        regions = self.detector.detect(frame)[: self.cfg.max_text_regions_per_frame]

        for region_index, region in enumerate(regions):
            crop = warp_text_region(frame, region.polygon)
            if crop is None:
                continue

            text, ocr_confidence = self.ocr.recognize(crop)
            if not text or ocr_confidence < self.cfg.min_confidence:
                continue

            # crop тут — уже геометрично виправлений (perspective-корекція) регіон;
            # саме його (за потреби) зберігає event_logger як покадровий "сирий" скріншот.
            event_logger.log(
                frame_index=frame_index,
                video_time_sec=video_time,
                text=text,
                detector_confidence=region.confidence,
                ocr_confidence=ocr_confidence,
                polygon=region.polygon,
                crop=crop,
                region_index=region_index,
            )

            if self.cfg.draw_annotations:
                pts = region.polygon.astype(int)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
