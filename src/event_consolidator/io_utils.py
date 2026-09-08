import json
from pathlib import Path
from typing import List

import numpy as np

from .models import RawEvent

def read_raw_events(path: Path) -> List[RawEvent]:
    events: List[RawEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                events.append(
                    RawEvent(
                        frame_index=data["frame_index"],
                        video_time_sec=data["video_time_sec"],
                        text=data["text"],
                        detector_confidence=float(data.get("detector_confidence", 0.0)),
                        ocr_confidence=float(data.get("ocr_confidence", 0.0)),
                        bbox=np.array(data["bbox"], dtype=np.float32),
                        crop_path=data.get("crop_path"),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Некоректний рядок {line_no} у {path}: {exc}") from exc
    return events

def write_consolidated_events(path: Path, events: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
