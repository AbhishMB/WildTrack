"""Live simulation: the real laptop app and dashboard against a simulated aircraft.

A fake phone serves the same WebSocket protocol as the bridge app (LinkServer.kt) on
localhost, and the synthetic camera renders frames delivered after the simulated video
lag. Open the dashboard, click the animal, ENGAGE, and watch the follow; the sim
panel's Pause button acts like the remote's Pause.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
import time
from collections import deque

import numpy as np

from ..app import App, SimHooks
from ..config import Config, override
from ..models import Detection, LatLon
from ..video_in import Frame, FrameSlot, StreamStats
from .camera import OracleTracker, SimCamera, SimDet
from .drone import SimPhone, start_drone
from .world import make_world

log = logging.getLogger(__name__)


class OracleDetector:
    """Detections straight from the simulator's ground truth (with noise), ByteTrack-like IDs."""

    def __init__(self) -> None:
        self.tracker = OracleTracker()

    def detect(self, frame: Frame, frame_t: float, tile: bool) -> list[Detection]:
        sim_dets: list[SimDet] = frame.meta or []
        ids = self.tracker.assign([s.animal_id for s in sim_dets])
        return [Detection(frame_t, ids[s.animal_id], s.cls, s.conf, s.box) for s in sim_dets]


class LiveSim:
    def __init__(self, cfg: Config):
        s = self.s = cfg.sim
        self.cfg = cfg
        self.rng = np.random.default_rng(s.seed)  # physics, event loop only
        self.cam_rng = np.random.default_rng(s.seed + 1)  # camera thread only
        self.world = make_world(s, self.rng, trees=True)
        self.drone = start_drone(s, self.world.target.n, self.world.target.e)
        self.home = LatLon(*s.home)
        self.phone = SimPhone(s, self.drone, self.home, cfg.geo.earth_radius_m)
        self.cam = SimCamera(s, cfg.camera.hfov_deg)
        self.lock = threading.Lock()
        self.slot = FrameSlot()
        self.stats = StreamStats()
        self.client = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------ phone side

    async def _send(self, msgs: list[dict]) -> None:
        ws = self.client
        if ws is None:
            return
        for m in msgs:
            try:
                await ws.send(json.dumps(m))
            except Exception:
                return

    async def _handler(self, ws) -> None:
        self.client = ws
        log.info("sim phone: laptop connected")
        try:
            async for raw in ws:
                msg = json.loads(raw)
                with self.lock:
                    events = self.phone.handle(msg, time.monotonic())
                await self._send(events)
        finally:
            if self.client is ws:
                self.client = None

    async def physics(self) -> None:
        dt = 0.01
        t_prev = time.monotonic()
        next_tel = t_prev
        while True:
            await asyncio.sleep(dt)
            t = time.monotonic()
            with self.lock:
                step = min(t - t_prev, 0.05)
                self.world.step(step, self.rng)
                self.drone.step(t, step)
                events = self.phone.check(t)
                tel = self.phone.telemetry() if t >= next_tel else None
            t_prev = t
            await self._send(events)
            if tel is not None:
                next_tel += 0.1
                await self._send([tel])

    def pause(self) -> dict:
        with self.lock:
            events = self.phone.pause()
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._send(events), self.loop)
        return {"ok": bool(events), "message": "pilot pressed Pause" if events else "Virtual Stick was not on"}

    # ------------------------------------------------------------ camera side

    def _render_loop(self) -> None:
        period = 1.0 / self.s.fps
        pending: deque[tuple[float, Frame]] = deque()
        idx = 0
        next_t = time.monotonic()
        self.cam.texture(self.s.seed)  # build the ground texture before the first frame
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_t:
                next_t += period
                with self.lock:
                    drone = copy.copy(self.drone)
                    world = copy.deepcopy(self.world)
                try:
                    img = self.cam.render(drone, world, self.s.seed)
                    dets = self.cam.detect(drone, world, self.cam_rng)
                except Exception as exc:  # e.g. the machine is out of memory: skip this frame, keep going
                    log.warning("sim camera frame skipped: %s", exc)
                    time.sleep(0.5)
                    continue
                pending.append((now + self.s.video_lag_s, Frame(idx, 0.0, img, dets)))
                idx += 1
            while pending and pending[0][0] <= time.monotonic():
                _, f = pending.popleft()
                t = time.monotonic()
                self.stats.connected = True
                self.stats.frame(t, f.image.shape[1], f.image.shape[0])
                self.slot.put(Frame(f.idx, t, f.image, f.meta))
            wake = min(next_t, pending[0][0] if pending else next_t)
            time.sleep(max(0.0, min(0.01, wake - time.monotonic())))

    async def serve(self) -> None:
        from websockets.asyncio.server import serve

        self.loop = asyncio.get_running_loop()
        threading.Thread(target=self._render_loop, name="sim-camera", daemon=True).start()
        async with serve(self._handler, "127.0.0.1", self.s.ws_port, ping_interval=None):
            await self.physics()


def sim_config(cfg: Config, auto: bool) -> Config:
    """Point the app at the simulator, with the lag 'calibrated' to the sim's true lag."""
    cfg = override(cfg, "link.backend", "dji")  # the simulator speaks the phone relay protocol
    cfg = override(cfg, "link.phone_host", "127.0.0.1")
    cfg = override(cfg, "link.ws_port", cfg.sim.ws_port)
    cfg = override(cfg, "video.lag_s", cfg.sim.video_lag_s)
    if auto:
        cfg = override(cfg, "lock.auto_lock", True)
        cfg = override(cfg, "lock.auto_engage", True)
    return cfg


async def run_live(cfg: Config, detector_kind: str, auto: bool) -> None:
    cfg = sim_config(cfg, auto)
    sim = LiveSim(cfg)
    if detector_kind == "yolo":
        from ..detect import YoloDetector

        detector = YoloDetector(cfg.detector, cfg.path(cfg.detector.weights))
    else:
        detector = OracleDetector()
    app = App(cfg, sim.slot, detector, sim.stats, sim=SimHooks(pause=sim.pause))
    await app.run(extra=[sim.serve()])
