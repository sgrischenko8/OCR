import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .geometry import average_polygon, iou
from .models import RawEvent, Track
from .text_similarity import text_similarity

logger = logging.getLogger(__name__)

_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\s]+')

def _sanitize_for_filename(text: str, max_len: int = 40) -> str:
    """Робить з розпізнаного тексту безпечний шматок імені файлу (без / \\ : * ? та пробілів)."""
    safe = _UNSAFE_FILENAME_CHARS.sub("_", text.strip()).strip("_")
    return (safe or "text")[:max_len]

class EventConsolidator:
    """
    Об'єднує покадрові записи event-логера в треки: одна й та ж фраза, що
    тримається на екрані кілька кадрів поспіль, стає одним підсумковим
    записом замість N окремих.

    Дві сусідні події вважаються продовженням одного треку, якщо:
      1) різниця кадрів у межах допустимого розриву (max_frame_gap);
      2) текст достатньо схожий (Левенштейн, а не точне ==) — щоб OCR-шум
         типу ABC-123 / ABC-128 не створював два різних треки;
      3) полігони достатньо перекриваються (IoU) — щоб два різних написи
         зі схожим текстом в різних місцях кадру не злиплись в один.

    Матчинг — жадібний (найкращі пари першими), не Hungarian: для типової
    кількості одночасних текстових регіонів на кадр цього достатньо.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        iou_threshold: float = 0.3,
        max_frame_gap: int = 0,
        min_run_length: int = 2,
        screenshots_dir: Optional[Path] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.iou_threshold = iou_threshold
        self.max_frame_gap = max_frame_gap
        self.min_run_length = min_run_length

        # Якщо задано — для кожного треку, що пройшов у фінальний результат,
        # з усіх його покадрових "сирих" кропів (crop_path у RawEvent) обирається
        # один найкращий (найвища ocr_confidence) і копіюється сюди. Тобто
        # скріншот з'являється лише для записів events_consolidated.jsonl,
        # а не для кожного кадру з events.jsonl.
        self.screenshots_dir = Path(screenshots_dir) if screenshots_dir else None
        if self.screenshots_dir is not None:
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self._active_tracks: List[Track] = []
        self._finished_tracks: List[Track] = []

    def consolidate(self, raw_events: List[RawEvent]) -> List[dict]:
        events_by_frame = self._group_by_frame(raw_events)

        for frame_index in sorted(events_by_frame.keys()):
            self._close_stale_tracks(frame_index)
            self._match_frame(events_by_frame[frame_index])

        self._close_stale_tracks(frame_index=None)  # закриваємо все, що лишилось активним

        return [
            self._build_output(track)
            for track in self._finished_tracks
            if len(track.observations) >= self.min_run_length
        ]

    @staticmethod
    def _group_by_frame(raw_events: List[RawEvent]) -> Dict[int, List[RawEvent]]:
        grouped: Dict[int, List[RawEvent]] = {}
        for event in raw_events:
            grouped.setdefault(event.frame_index, []).append(event)
        return grouped

    def _close_stale_tracks(self, frame_index: Optional[int]):
        """Закриває треки, чий розрив до frame_index перевищив ліміт (або взагалі всі, якщо frame_index=None)."""
        still_active = []
        for track in self._active_tracks:
            gap_exceeded = frame_index is None or (frame_index - track.last_frame - 1) > self.max_frame_gap
            if gap_exceeded:
                self._finished_tracks.append(track)
            else:
                still_active.append(track)
        self._active_tracks = still_active

    def _match_frame(self, frame_events: List[RawEvent]):
        candidates = []  # (score, track, event)
        for track in self._active_tracks:
            for event in frame_events:
                similarity = text_similarity(track.last_text, event.text)
                overlap = iou(track.last_bbox, event.bbox)
                if similarity >= self.similarity_threshold and overlap >= self.iou_threshold:
                    candidates.append((similarity * overlap, track, event))

        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_track_ids, matched_event_ids = set(), set()
        for _, track, event in candidates:
            if id(track) in matched_track_ids or id(event) in matched_event_ids:
                continue
            track.observations.append(event)
            matched_track_ids.add(id(track))
            matched_event_ids.add(id(event))

        # усе, що лишилось незматченим у кадрі — старт нового треку
        for event in frame_events:
            if id(event) not in matched_event_ids:
                self._active_tracks.append(Track(observations=[event]))

    def _build_output(self, track: Track) -> dict:
        observations = track.observations
        output = {
            "text": self._pick_canonical_text(observations),
            "frame_start": observations[0].frame_index,
            "frame_end": observations[-1].frame_index,
            "frame_count": len(observations),
            "time_start_sec": round(observations[0].video_time_sec, 3),
            "time_end_sec": round(observations[-1].video_time_sec, 3),
            "detector_confidence": round(
                sum(o.detector_confidence for o in observations) / len(observations), 3
            ),
            "ocr_confidence": round(
                sum(o.ocr_confidence for o in observations) / len(observations), 3
            ),
            "polygon": average_polygon([o.bbox for o in observations]).round(1).tolist(),
        }

        screenshot_path = self._save_track_screenshot(observations, output)
        if screenshot_path is not None:
            output["screenshot"] = screenshot_path

        return output

    def _save_track_screenshot(self, observations: List[RawEvent], output: dict) -> Optional[str]:
        """
        Обирає з покадрових кропів треку один найкращий (найвища ocr_confidence
        серед кадрів з існуючим crop_path) і копіює його у screenshots_dir.
        Це єдина точка, де "сирі" покадрові скріншоти (по одному на кожен
        запис events.jsonl) перетворюються на один скріншот на консолідовану
        подію events_consolidated.jsonl.
        """
        if self.screenshots_dir is None:
            return None

        candidates = [o for o in observations if o.crop_path and Path(o.crop_path).exists()]
        if not candidates:
            return None

        best = max(candidates, key=lambda o: o.ocr_confidence)

        safe_text = _sanitize_for_filename(output["text"])
        filename = f"{output['frame_start']:07d}-{output['frame_end']:07d}_{safe_text}.png"
        dest_path = self.screenshots_dir / filename

        try:
            shutil.copyfile(best.crop_path, dest_path)
        except OSError:
            logger.exception("Не вдалося скопіювати скріншот треку у %s", dest_path)
            return None

        return str(dest_path)

    @staticmethod
    def _pick_canonical_text(observations: List[RawEvent]) -> str:
        """
        OCR трохи "тремтить" по кадрах (ABC-123 / ABC-128 і т.п.), тому замість
        тексту останнього кадру беремо найчастіший варіант у треку, а при
        рівності кількості — з найбільшою сумарною OCR-впевненістю.
        """
        groups: Dict[str, Dict[str, float]] = {}
        for obs in observations:
            group = groups.setdefault(obs.text, {"count": 0, "conf_sum": 0.0})
            group["count"] += 1
            group["conf_sum"] += obs.ocr_confidence

        best_text, _ = max(groups.items(), key=lambda kv: (kv[1]["count"], kv[1]["conf_sum"]))
        return best_text
