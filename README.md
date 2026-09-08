# video-text-pipeline

A pipeline for detecting and recognizing text in video, with subsequent
consolidation of per-frame results into phrase-level records.

```
Video -> Text Detection -> Perspective Correction -> OCR -> Event Logger -> Consolidator
```

No video is displayed on screen (headless). Progress percentage is written
to the log during processing. On Ctrl+C the program exits cleanly — the
output video won't be corrupted.

## Project structure

```
main.py                              # CLI: runs the pipeline, then consolidation
run_consolidator.py                  # CLI: consolidation only (can be run standalone)

src/video_text_pipeline/
  config.py                          # PipelineConfig — pipeline settings
  video_io.py                        # VideoReader / VideoWriter wrappers
  perspective.py                     # perspective correction of a text region
  ocr.py                             # wrapper over easyocr.Reader.recognize()
  event_logger.py                    # per-frame event logging to JSONL
  pipeline.py                        # glues all stages together
  detectors/
    base.py                          # detector interface
    easyocr_detector.py              # default detector — easyocr's CRAFT model
    mser_detector.py                 # lightweight fallback with no weights to download
    east_detector.py                 # EAST-based detector (needs a separate model file)

src/event_consolidator/
  models.py                          # RawEvent, Track
  text_similarity.py                 # normalized string similarity (Levenshtein)
  geometry.py                        # polygon IoU, averaging a track's position
  tracker.py                         # EventConsolidator — logic for merging frames into tracks
  io_utils.py                        # JSONL reading/writing
```

## Installation

```bash
pip install -e .
```

All dependencies (opencv-python-headless, numpy, easyocr) are listed in
`pyproject.toml` and are installed automatically together with the package.

No separate system binary is needed — `easyocr` downloads model weights
itself on first run (internet is needed once, then it works offline).

For the `--detector east` backend, download
`frozen_east_text_detection.pb` separately and pass its path via
`--east-model`.

## Running

Full pipeline (detection + OCR + consolidation in a single call):

```bash
python main.py input.mp4 output.mp4 --log events.jsonl
```

By default OCR recognizes English (`--ocr-lang en`). For other languages —
comma-separated codes, e.g. `--ocr-lang en,ru`.

Consolidation only, on an already-existing `events.jsonl` (without
reprocessing the video):

```bash
python run_consolidator.py events.jsonl events_consolidated.jsonl \
  --min-run 2 --max-gap 0 --text-similarity 0.7 --iou 0.3
```

## Consolidation logic

Two neighboring per-frame events are considered a continuation of the same
phrase if:

1. the frames follow without a gap (or within `--max-gap`);
2. the text is similar enough by normalized Levenshtein distance
   (`--text-similarity`, default 0.7) — so that "ABC-123" and "ABC-128"
   are treated as one phrase with OCR noise rather than two different ones;
3. the polygons overlap enough (`--iou`, default 0.3) — so that different
   text in the same place on the frame doesn't get merged into one phrase
   just because the strings happen to match.

Matching is greedy (best pairs first), not Hungarian: for the typical
number of simultaneous text regions per frame this is enough.
A track that gathers fewer than `--min-run` frames (default 2) is
discarded as detector noise.

## events.jsonl format (per-frame, from event_logger.py)

```json
{"frame_index": 142, "video_time_sec": 4.733, "text": "STOP",
 "detector_confidence": 0.81, "ocr_confidence": 0.93,
 "bbox": [[120.0, 40.0], [210.0, 42.0], [208.0, 70.0], [118.0, 68.0]],
 "bbox_center": [164.0, 55.0]}
```

## events_consolidated.jsonl format (after consolidation)

```json
{
  "text": "ABC-123",
  "frame_start": 120,
  "frame_end": 123,
  "frame_count": 4,
  "time_start_sec": 4.0,
  "time_end_sec": 4.1,
  "detector_confidence": 1.0,
  "ocr_confidence": 0.805,
  "polygon": [[101.0, 50.2], [151.0, 50.2], [151.0, 70.2], [101.0, 70.2]]
}
```

`detector_confidence` and `ocr_confidence` are simple averages across all
frames of the track, with no additional metrics like "stability" — that's
enough for the typical use case.

## Screenshots of detected text

For each consolidated event (`events_consolidated.jsonl`) a screenshot of
the detected text can be saved — the actual crop that already went through
perspective correction (the same one that's fed into OCR), not the raw
frame as a whole.

This works in two steps:

1. During the main video processing (`pipeline.py`/`main.py`) every
   per-frame crop (for each line of `events.jsonl`) is additionally saved
   to a temporary folder (by default `<--log>_crops/`), and its path is
   written into `events.jsonl` itself as a `crop_path` field.
2. The consolidator (`run_consolidator.py`), while building a track,
   picks **one** crop out of all its per-frame crops — the one with the
   highest `ocr_confidence` — and copies it into the final screenshots
   folder (by default `<output_log>_screenshots/`), writing the path into
   the `screenshot` field of the corresponding `events_consolidated.jsonl`
   record.

In other words, screenshots are saved **only** for events that made it
into `events_consolidated.jsonl` — single-frame/noisy detections that
didn't form a track (`--min-run`) get no screenshot. After consolidation,
the temporary per-frame crops folder (step 1) is automatically removed —
only the final screenshots remain.

Controlled via flags (available in both `main.py` and `run_consolidator.py`):

```bash
python main.py input.mp4 output.mp4 --log events.jsonl \
  --screenshots-dir screenshots \
  # --no-screenshots     — don't save screenshots at all
  # --keep-raw-crops     — don't delete per-frame crops after consolidation

python run_consolidator.py events.jsonl events_consolidated.jsonl \
  --screenshots-dir screenshots
```

## Deliberate architectural decisions (not bugs)

- **OCR runs on every frame, with no frame skipping and no lightweight
  tracker (KCF / optical flow) between OCR calls.** For tasks where the
  priority is maximum recognition reliability rather than processing
  speed (an offline pipeline with no real-time requirement), this is a
  deliberate choice: every frame is guaranteed to get a fresh detector and
  OCR result, with no risk of "losing" text between skipped frames due to
  tracker drift. The cost is a higher total processing time on long
  videos. If speed becomes a priority, the simplest next step is to run
  the full detector+OCR once every N frames and update the bbox via
  optical flow in between.
- **Track matching is greedy, not Hungarian** — deliberately simplified,
  because the number of simultaneous text regions per frame in the target
  scenarios is small.
- **No GUI/`cv2.imshow`** — the pipeline is fully headless, which is why
  `pyproject.toml` intentionally depends on `opencv-python-headless`
  rather than `opencv-python`.
- **`main.py` always runs consolidation in `finally`, even if video
  processing crashed with an exception partway through.** This is
  deliberate: if the pipeline crashed on frame N out of M, `events.jsonl`
  already contains all the per-frame events accumulated before the
  crash (`EventLogger` writes in a streaming fashion, without buffering)
  — so it makes sense to consolidate at least that part rather than
  discard all progress. Video-processing and consolidation errors are
  logged separately via `logging.exception(...)`, with a full traceback,
  so that a partial log is clearly visible in the logs rather than hidden
  behind a short `str(e)`.
