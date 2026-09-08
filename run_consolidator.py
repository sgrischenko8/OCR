import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional, Set

from event_consolidator.io_utils import read_raw_events, write_consolidated_events
from event_consolidator.tracker import EventConsolidator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Об'єднує покадровий event-лог у записи по фразах (мінімум N кадрів поспіль)"
    )
    parser.add_argument(
        "input_log",
        type=Path,
        nargs="?",
        default=Path("events.jsonl"),
        help="Вхідний JSONL з event_logger.py (за замовчуванням: events.jsonl)",
    )
    parser.add_argument(
        "output_log",
        type=Path,
        nargs="?",
        default=Path("events_consolidated.jsonl"),
        help="Куди зберегти консолідований JSONL (за замовчуванням: events_consolidated.jsonl)",
    )
    parser.add_argument("--min-run", type=int, default=2, help="Мінімум кадрів поспіль, щоб фраза потрапила в результат")
    parser.add_argument("--max-gap", type=int, default=0, help="Скільки кадрів пропуску всередині треку ще дозволено")
    parser.add_argument("--text-similarity", type=float, default=0.7, help="Поріг схожості тексту 0..1 (Левенштейн)")
    parser.add_argument("--iou", type=float, default=0.3, help="Поріг перетину bbox 0..1 для зв'язування кадрів")
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=None,
        help="Куди зберігати скріншоти консолідованих подій (за замовчуванням: '<output_log>_screenshots' поруч із output_log)",
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Не зберігати скріншоти для events_consolidated.jsonl",
    )
    parser.add_argument(
        "--keep-raw-crops",
        action="store_true",
        help="Не видаляти покадрові 'сирі' кропи (з events.jsonl) після консолідації",
    )
    return parser.parse_args()

def run_consolidation(
    input_log: Path = Path("events.jsonl"),
    output_log: Path = Path("events_consolidated.jsonl"),
    min_run: int = 2,
    max_gap: int = 0,
    text_similarity: float = 0.7,
    iou: float = 0.3,
    save_screenshots: bool = True,
    screenshots_dir: Optional[Path] = None,
    cleanup_raw_crops: bool = True,
):
    print("[INFO] Запуск консолідації подій...")

    if not input_log.exists():
        logging.warning("Файл логів %s не знайдено. Консолідацію скасовано.", input_log)
        return

    raw_events = read_raw_events(input_log)
    logging.info("Прочитано %d покадрових подій з %s", len(raw_events), input_log)

    resolved_screenshots_dir = None
    if save_screenshots:
        resolved_screenshots_dir = (
            Path(screenshots_dir)
            if screenshots_dir is not None
            else output_log.parent / f"{output_log.stem}_screenshots"
        )

    consolidator = EventConsolidator(
        similarity_threshold=text_similarity,
        iou_threshold=iou,
        max_frame_gap=max_gap,
        min_run_length=min_run,
        screenshots_dir=resolved_screenshots_dir,
    )
    consolidated = consolidator.consolidate(raw_events)
    consolidated.sort(key=lambda e: e["frame_start"])

    write_consolidated_events(output_log, consolidated)

    raw_count = len(raw_events)
    used_count = sum(e.get("frame_count", 0) for e in consolidated)
    screenshots_count = sum(1 for e in consolidated if e.get("screenshot"))
    logging.info(
        "Записано %d консолідованих подій у %s (відкинуто одиночних/коротких треків: %d)",
        len(consolidated),
        output_log,
        raw_count - used_count,
    )
    if resolved_screenshots_dir is not None:
        logging.info(
            "Збережено %d скріншотів (по одному на консолідовану подію) у %s",
            screenshots_count,
            resolved_screenshots_dir,
        )

    if cleanup_raw_crops:
        _cleanup_raw_crop_dirs(raw_events)

def _cleanup_raw_crop_dirs(raw_events) -> None:
    """
    Видаляє тимчасові папки з покадровими "сирими" кропами (crop_path з
    events.jsonl) — вони більше не потрібні після того, як потрібні кадри
    вже скопійовані у screenshots_dir для events_consolidated.jsonl.
    """
    raw_crop_dirs: Set[Path] = {
        Path(e.crop_path).parent for e in raw_events if e.crop_path
    }
    for crop_dir in raw_crop_dirs:
        if crop_dir.exists():
            shutil.rmtree(crop_dir, ignore_errors=True)
            logging.info("Видалено тимчасову папку покадрових кропів: %s", crop_dir)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    run_consolidation(
        input_log=args.input_log,
        output_log=args.output_log,
        min_run=args.min_run,
        max_gap=args.max_gap,
        text_similarity=args.text_similarity,
        iou=args.iou,
        save_screenshots=not args.no_screenshots,
        screenshots_dir=args.screenshots_dir,
        cleanup_raw_crops=not args.keep_raw_crops,
    )

if __name__ == "__main__":
    main()
