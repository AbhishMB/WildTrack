"""YOLO26 + ByteTrack, and the worker thread that runs it on the newest frame.

The spec's `model.track(frame, persist=True, tracker="bytetrack.yaml")` is what this
does internally; here the tracker is held explicitly so that one ByteTrack instance is
fed by either full-frame inference or SAHI-style tiles (Manual search only), and track
IDs stay continuous when tiling switches on or off.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .config import DetectorCfg
from .models import Detection
from .video_in import Frame, FrameSlot

log = logging.getLogger(__name__)


class DetectorLike(Protocol):
    def detect(self, frame: Frame, frame_t: float, tile: bool) -> list[Detection]: ...


def tile_origins(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    """Top-left corners of overlapping tiles covering the frame."""
    step = max(1, int(tile * (1.0 - overlap)))

    def axis(n: int) -> list[int]:
        if n <= tile:
            return [0]
        xs = list(range(0, n - tile, step)) + [n - tile]
        return sorted(set(xs))

    return [(x, y) for y in axis(height) for x in axis(width)]


def touches_inner_edge(b: np.ndarray, x: int, y: int, tile: int, width: int, height: int, margin: float = 2.0) -> np.ndarray:
    """Boxes cut by a tile edge that is not the image edge. The overlapping neighbour
    tile (or the full-frame pass, for big animals) sees those animals whole."""
    if len(b) == 0:
        return np.zeros(0, bool)
    tw, th = min(tile, width - x), min(tile, height - y)
    cut = np.zeros(len(b), bool)
    if x > 0:
        cut |= b[:, 0] <= margin
    if y > 0:
        cut |= b[:, 1] <= margin
    if x + tw < width:
        cut |= b[:, 2] >= tw - margin
    if y + th < height:
        cut |= b[:, 3] >= th - margin
    return cut


def nms(boxes: np.ndarray, iou: float) -> np.ndarray:
    """Class-wise NMS on rows [x1, y1, x2, y2, conf, cls]; returns kept rows."""
    if len(boxes) == 0:
        return boxes
    keep = []
    for c in np.unique(boxes[:, 5]):
        b = boxes[boxes[:, 5] == c]
        b = b[np.argsort(-b[:, 4])]
        while len(b):
            keep.append(b[0])
            if len(b) == 1:
                break
            xx1 = np.maximum(b[0, 0], b[1:, 0])
            yy1 = np.maximum(b[0, 1], b[1:, 1])
            xx2 = np.minimum(b[0, 2], b[1:, 2])
            yy2 = np.minimum(b[0, 3], b[1:, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])  # noqa: E731
            ov = inter / (area(b[0]) + area(b[1:]) - inter + 1e-9)
            b = b[1:][ov < iou]
    return np.array(keep).reshape(-1, 6)


class YoloDetector:
    def __init__(self, cfg: DetectorCfg, weights: Path):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(str(weights))
        self.names: dict[int, str] = dict(self.model.names)
        half = cfg.half and not cfg.device.startswith("cpu")
        self.precision = self._precision_args(half)
        self.tracker = self._make_tracker(cfg.tracker)
        log.info("detector %s on %s, classes %s", weights, cfg.device, list(self.names.values()))

    @staticmethod
    def _precision_args(half: bool) -> dict:
        """FP16 on GPU. Newer ultralytics spells it quantize=16; older ones half=True."""
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16 if half else None}
        return {"half": half}

    @staticmethod
    def _make_tracker(tracker_yaml: str):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        args = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))
        try:
            return BYTETracker(args=args)
        except TypeError:  # ultralytics < 8.4 also wants the frame rate
            return BYTETracker(args=args, frame_rate=30)

    def reset(self) -> None:
        self.tracker.reset()

    def _predict(self, imgs: list[np.ndarray], imgsz: int) -> list[np.ndarray]:
        res = self.model.predict(imgs, imgsz=imgsz, conf=self.cfg.conf, iou=self.cfg.iou,
                                 device=self.cfg.device, verbose=False, **self.precision)
        return [r.boxes.data.cpu().numpy().reshape(-1, 6) for r in res]

    def _tiled(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        ts = self.cfg.tile_size
        origins = tile_origins(w, h, ts, self.cfg.tile_overlap)
        crops = [img[y : y + ts, x : x + ts] for x, y in origins]
        parts = [self._predict([img], self.cfg.imgsz)[0]]  # full frame keeps large animals whole
        for (x, y), b in zip(origins, self._predict(crops, ts)):
            b = b[~touches_inner_edge(b, x, y, ts, w, h)]
            if len(b):
                b = b.copy()
                b[:, [0, 2]] += x
                b[:, [1, 3]] += y
                parts.append(b)
        return nms(np.vstack(parts), self.cfg.iou)

    def detect(self, frame: Frame, frame_t: float, tile: bool) -> list[Detection]:
        from ultralytics.engine.results import Boxes

        img = frame.image
        raw = self._tiled(img) if tile else self._predict([img], self.cfg.imgsz)[0]
        tracks = self.tracker.update(Boxes(raw, img.shape[:2]), img)
        out = []
        for row in tracks:
            x1, y1, x2, y2, tid, score, cls = row[:7]
            out.append(Detection(frame_t, int(tid), self.names.get(int(cls), str(int(cls))), float(score),
                                 (float(x1), float(y1), float(x2), float(y2))))
        return out


@dataclass
class DetResult:
    frame: Frame
    dets: list[Detection]
    infer_ms: float


class DetectWorker:
    """Runs detection in its own thread so it never blocks the 10 Hz command loop."""

    def __init__(
        self,
        slot: FrameSlot,
        detector: DetectorLike,
        lag_s: float,
        tile_now: Callable[[], bool],
        on_result: Callable[[DetResult], None],
    ):
        self.slot = slot
        self.detector = detector
        self.lag_s = lag_s
        self.tile_now = tile_now
        self.on_result = on_result
        self.infer_ms = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="detect", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        idx = -1
        while not self._stop.is_set():
            f = self.slot.wait_newer(idx, 0.5)
            if f is None:
                continue
            idx = f.idx
            t0 = time.perf_counter()
            try:
                dets = self.detector.detect(f, f.t_arrival - self.lag_s, self.tile_now())
            except Exception:
                log.exception("detector failed on frame %d", f.idx)
                continue
            self.infer_ms = (time.perf_counter() - t0) * 1000.0
            try:
                self.on_result(DetResult(f, dets, self.infer_ms))
            except Exception:
                log.exception("detection consumer failed")
