import argparse
import logging
import signal
from pathlib import Path

from video_text_pipeline.config import PipelineConfig
from video_text_pipeline.pipeline import VideoTextPipeline
from run_consolidator import run_consolidation

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пайплайн детекції та розпізнавання тексту на відео")
    parser.add_argument("input", type=Path, nargs="?", default=Path("input.mp4"), help="Шлях до вхідного відео (за замовчуванням: input.mp4)")
    parser.add_argument("output", type=Path, nargs="?", default=Path("output.mp4"), help="Шлях для збереження обробленого відео (за замовчуванням: output.mp4)")
    parser.add_argument("--log", type=Path, default=Path("events.jsonl"), help="Файл для лога подій (JSONL)")
    parser.add_argument("--detector", choices=["easyocr", "mser", "east"], default="easyocr", help="Бекенд детекції тексту")
    parser.add_argument("--east-model", type=Path, default=None, help="Шлях до frozen_east_text_detection.pb")
    parser.add_argument("--ocr-lang", default="en", help="Мови для easyocr через кому, напр. 'en,ru'")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Поріг впевненості детектора (mser/east)")
    parser.add_argument("--gpu", action="store_true", help="Використати CUDA для easyocr, якщо доступна")
    parser.add_argument("--no-annotations", action="store_true", help="Не малювати рамки на вихідному відео")
    parser.add_argument(
        "--consolidated-log",
        type=Path,
        default=Path("events_consolidated.jsonl"),
        help="Файл для консолідованих подій (за замовчуванням: events_consolidated.jsonl)",
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Не зберігати скріншоти знайдених текстів для консолідованих подій",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=None,
        help="Куди зберігати скріншоти консолідованих подій (за замовчуванням: '<consolidated-log>_screenshots')",
    )
    parser.add_argument(
        "--keep-raw-crops",
        action="store_true",
        help="Не видаляти покадрові 'сирі' кропи (з --log) після консолідації",
    )
    return parser.parse_args()

def run_video_pipeline(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        input_video=args.input,
        output_video=args.output,
        log_path=args.log,
        detector_type=args.detector,
        east_model_path=args.east_model,
        ocr_lang=args.ocr_lang,
        min_confidence=args.min_confidence,
        use_gpu=args.gpu,
        draw_annotations=not args.no_annotations,
        save_crops=not args.no_screenshots,
    )

    pipeline = VideoTextPipeline(cfg)

    def handle_sigint(signum, frame):
        logging.info("Отримано SIGINT (Ctrl+C), завершуємо обробку відео...")
        pipeline.request_stop()

    signal.signal(signal.SIGINT, handle_sigint)

    pipeline.run()
    return cfg

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = None
    try:
        cfg = run_video_pipeline(args)
    except Exception:
        logging.exception("Помилка під час обробки відео")
    finally:
        try:
            print("\n[INFO] Запуск консолидайції подій...")
            # events.jsonl пишеться стрімінгово (без буферизації), тож навіть
            # якщо обробка відео впала посередині — консолідуємо те, що встигло
            # накопичитись, і зберігаємо скріншоти саме для цих подій.
            run_consolidation(
                input_log=cfg.log_path if cfg is not None else args.log,
                output_log=args.consolidated_log,
                save_screenshots=not args.no_screenshots,
                screenshots_dir=args.screenshots_dir,
                cleanup_raw_crops=not args.keep_raw_crops,
            )
        except Exception:
            logging.exception("Помилка під час консолідації")

if __name__ == "__main__":
    main()