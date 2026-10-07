"""Per-sortie log for replay.

logs/<YYYYmmdd-HHMMSS>/
    meta.json      config snapshot, wall-clock offset, versions
    events.jsonl   one JSON object per line, each with "t" = laptop monotonic seconds
    video.mp4      the frames detection ran on; "frame" lines in events.jsonl give their times
    animal.gpx / animal.kml   written when the sortie closes

Writing happens in a background thread so logging never blocks the command loop.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _jsonable(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (np.floating, np.integer)):
        return _jsonable(x.item())
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(dataclasses.asdict(x))
    if isinstance(x, Path):
        return str(x)
    return x


class SortieLogger:
    def __init__(self, root: Path, meta: dict, record_video: bool, video_fps: float):
        self.dir = Path(root) / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.wall_offset_s = time.time() - time.monotonic()
        meta = dict(meta, wall_offset_s=self.wall_offset_s, started=datetime.now().isoformat())
        (self.dir / "meta.json").write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
        self._q: queue.Queue = queue.Queue(maxsize=5000)
        self._record_video = record_video
        self._video_fps = video_fps
        self._writer = None
        self._size = (0, 0)
        self._frame_idx = 0
        self._frame_lock = threading.Lock()
        self._dropped = 0
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="logger", daemon=True)
        self._thread.start()
        log.info("sortie log: %s", self.dir)

    def __call__(self, kind: str, t: float, **fields: Any) -> None:
        self.event(kind, t, **fields)

    def event(self, kind: str, t: float, **fields: Any) -> None:
        self._put(("event", {"t": t, "kind": kind, **fields}))

    def frame(self, t_arrival: float, frame_t: float, image: np.ndarray) -> int | None:
        """Queue a frame for video.mp4; returns its index in the video, or None if not recorded."""
        if not self._record_video:
            return None
        with self._frame_lock:
            idx = self._frame_idx
            if not self._put(("frame", (idx, t_arrival, frame_t, image))):
                return None
            self._frame_idx += 1
            return idx

    def _put(self, item) -> bool:
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def _run(self) -> None:
        import cv2

        while True:
            item = self._q.get()
            if item is None:
                break
            kind, payload = item
            try:
                if kind == "event":
                    self._events.write(json.dumps(_jsonable(payload)) + "\n")
                else:
                    idx, t_arrival, frame_t, img = payload
                    if self._writer is None:
                        h, w = img.shape[:2]
                        self._size = (w, h)
                        self._writer = cv2.VideoWriter(str(self.dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                       self._video_fps, (w, h))
                    if (img.shape[1], img.shape[0]) != self._size:
                        img = cv2.resize(img, self._size)  # stream resolution changed mid-sortie
                    self._writer.write(img)
                    rec = {"t": t_arrival, "kind": "frame", "idx": idx, "frame_t": frame_t}
                    self._events.write(json.dumps(rec) + "\n")
            except Exception:
                log.exception("log write failed")
            if self._q.empty():
                self._events.flush()

    def write_file(self, name: str, text: str) -> Path:
        p = self.dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=10)
        if self._writer is not None:
            self._writer.release()
        self._events.close()
        if self._dropped:
            log.warning("logger dropped %d items (disk too slow)", self._dropped)


def read_events(log_dir: Path) -> list[dict]:
    with open(Path(log_dir) / "events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
