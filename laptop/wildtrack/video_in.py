"""RTSP video in, newest frame only.

PyAV (FFmpeg) with low-delay options reads the phone's RTSP stream in its own thread.
Each decoded frame overwrites a one-frame slot, stamped with its laptop arrival time,
so the detector always works on the newest frame and frames are never stale.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import VideoCfg

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    idx: int
    t_arrival: float  # laptop monotonic
    image: np.ndarray  # BGR
    meta: Any = None  # simulator only: ground truth for the oracle detector


class FrameSlot:
    """One-frame mailbox: the reader overwrites, the detector takes the newest."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._frame: Frame | None = None

    def put(self, f: Frame) -> None:
        with self._cv:
            self._frame = f
            self._cv.notify_all()

    def newest(self) -> Frame | None:
        with self._cv:
            return self._frame

    def wait_newer(self, idx: int, timeout: float) -> Frame | None:
        with self._cv:
            self._cv.wait_for(lambda: self._frame is not None and self._frame.idx > idx, timeout)
            f = self._frame
            return f if f is not None and f.idx > idx else None


class StreamStats:
    def __init__(self) -> None:
        self.frames = 0
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.connected = False
        self.last_t = 0.0

    def frame(self, t: float, w: int, h: int) -> None:
        if self.last_t:
            dt = t - self.last_t
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1.0 / dt
        self.last_t, self.width, self.height = t, w, h
        self.frames += 1


class RtspReader:
    def __init__(self, cfg: VideoCfg, slot: FrameSlot, clock=time.monotonic):
        self.cfg = cfg
        self.slot = slot
        self.clock = clock
        self.stats = StreamStats()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rtsp", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _options(self) -> dict[str, str]:
        rtsp = {"rtsp_transport": self.cfg.rtsp_transport} if self.cfg.rtsp_url.startswith("rtsp") else {}
        return {
            **rtsp,
            "fflags": "nobuffer",
            "flags": "low_delay",
            "max_delay": "0",
            "reorder_queue_size": "0",
            "analyzeduration": "1000000",
            "probesize": "1000000",
        }

    def _run(self) -> None:
        import av

        idx = 0
        while not self._stop.is_set():
            container = None
            try:
                container = av.open(self.cfg.rtsp_url, options=self._options(),
                                    timeout=(self.cfg.open_timeout_s, self.cfg.open_timeout_s))
                stream = container.streams.video[0]
                try:
                    stream.codec_context.flags |= av.codec.context.Flags.low_delay
                except Exception:  # older PyAV: the container option above still applies
                    pass
                self.stats.connected = True
                log.info("video: %s %s", self.cfg.rtsp_url, stream.codec_context.name)
                for packet in container.demux(stream):
                    if self._stop.is_set():
                        break
                    for frame in packet.decode():
                        t = self.clock()
                        img = frame.to_ndarray(format="bgr24")
                        self.stats.frame(t, img.shape[1], img.shape[0])
                        self.slot.put(Frame(idx, t, img))
                        idx += 1
            except Exception as exc:
                if self.stats.connected:
                    log.warning("video lost: %s", exc)
                else:
                    log.debug("video connect failed: %s", exc)
            finally:
                self.stats.connected = False
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass
            self._stop.wait(self.cfg.reconnect_s)
