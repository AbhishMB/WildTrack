"""The laptop app: one asyncio process.

    video (thread) -> newest-frame slot -> detection (thread) --call_soon_threadsafe--> Autopilot
    phone link (WebSocket) --------------------------------------------------------> Autopilot
    10 Hz control loop: Autopilot.tick() -> phone link
    dashboard (FastAPI/uvicorn in the same loop) <-> browsers

Detection never blocks the command loop; the Autopilot is only touched from the loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import numpy as np

from . import __version__
from .config import Config
from .core import Autopilot
from .dashboard.server import Hub, TileStore, create_app, ui_config
from .dem import Terrain
from .detect import DetectorLike, DetectWorker, DetResult
from .export import to_gpx, to_kml
from .fsm import Mode
from .tello import make_link
from .logger import SortieLogger
from .models import Telemetry
from .video_in import FrameSlot, StreamStats

log = logging.getLogger(__name__)


@dataclass
class SimHooks:
    """Simulator-only controls the dashboard may call."""

    pause: Callable[[], dict]


def encode_jpeg(img: np.ndarray, max_width: int, quality: int) -> tuple[bytes, float]:
    import cv2

    h, w = img.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return (buf.tobytes() if ok else b""), scale


class App:
    def __init__(
        self,
        cfg: Config,
        slot: FrameSlot,
        detector: DetectorLike,
        video_stats: StreamStats,
        sim: SimHooks | None = None,
    ):
        self.cfg = cfg
        self.slot = slot
        self.detector = detector
        self.video_stats = video_stats
        self.sim = sim
        self.terrain = Terrain.from_dir(cfg.path(cfg.geo.dem_dir))
        meta = {"version": __version__, "config": dataclasses.asdict(cfg), "sim": sim is not None}
        self.logger = SortieLogger(cfg.path(cfg.logging.dir), meta, cfg.logging.record_video, cfg.logging.video_fps)
        self.ap = Autopilot(cfg, self.terrain, log=self.logger)
        self.hub = Hub(cfg.dashboard, ui_config(cfg.dashboard, {"sim": sim is not None, "lag_s": cfg.video.lag_s, "backend": cfg.link.backend}))
        self.hub.on_command = self.on_command
        self.tiles = TileStore(cfg.path(cfg.dashboard.tiles_dir),
                               cfg.path(cfg.dashboard.mbtiles) if cfg.dashboard.mbtiles else None)
        self.link = make_link(cfg, self._on_telemetry, self.ap.on_phone_event, self.ap.on_link)
        self.worker = DetectWorker(slot, detector, cfg.video.lag_s, self._tile_now, self._on_result_thread)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.infer_ms = 0.0
        self._alerts_sent = 0
        self._tx_count = 0

    # --------------------------------------------------------------- inputs

    def _on_telemetry(self, tel: Telemetry, msg: dict, t: float) -> None:
        self.logger("telemetry", t, msg=msg)
        self.ap.on_telemetry(tel, t)

    def _tile_now(self) -> bool:
        return self.cfg.detector.tile_search and self.ap.fsm.mode == Mode.MANUAL

    def _on_result_thread(self, res: DetResult) -> None:
        """Detection worker thread: record and encode here, then hand over to the loop."""
        f = res.frame
        vidx = self.logger.frame(f.t_arrival, f.t_arrival - self.cfg.video.lag_s, f.image)
        jpeg, scale = encode_jpeg(f.image, self.cfg.dashboard.video_max_width, self.cfg.dashboard.jpeg_quality)
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._on_result, res, vidx, jpeg, scale)

    def _on_result(self, res: DetResult, vidx: int | None, jpeg: bytes, scale: float) -> None:
        t = time.monotonic()
        f = res.frame
        h, w = f.image.shape[:2]
        self.infer_ms = res.infer_ms
        self.logger("dets", t, video_idx=vidx, frame_t=f.t_arrival - self.cfg.video.lag_s, infer_ms=res.infer_ms,
                    w=w, h=h, dets=[[d.track_id, d.cls, round(d.conf, 3), [round(x, 1) for x in d.box_xyxy]] for d in res.dets])
        self.ap.on_detections(res.dets, w, h, t)
        locks = self.ap.locks
        meta = {
            "type": "frame",
            "w": int(w * scale),
            "h": int(h * scale),
            "src_w": w,
            "src_h": h,
            "infer_ms": round(res.infer_ms, 1),
            "dets": [
                {
                    "id": d.track_id,
                    "cls": d.cls,
                    "conf": round(d.conf, 2),
                    "box": [round(x * scale, 1) for x in d.box_xyxy],
                    "lockable": locks.is_lockable(d.cls),
                    "candidate": locks.is_candidate(d),
                    "locked": d.track_id == locks.locked_id,
                }
                for d in res.dets
            ],
        }
        self.hub.publish_frame(meta, jpeg)

    async def on_command(self, action: str, msg: dict, control: bool) -> dict:
        t = time.monotonic()
        if action == "release":
            reply = self.ap.operator("release", t)
        elif action in ("lock", "engage"):
            if not control:
                return {"ok": False, "message": "view-only screen: LOCK and ENGAGE only from the ground-station laptop"}
            tid = msg.get("track_id")
            reply = self.ap.operator(action, t, int(tid) if tid is not None else None)
        elif action == "sim_pause" and self.sim is not None:
            reply = self.sim.pause()
        elif action in ("takeoff", "land", "emergency", "stick"):
            reply = await self._manual(action, msg, control)
            if reply is None:
                return None  # sticks are sent at 10 Hz: no reply per message
        else:
            return {"ok": False, "message": f"unknown action {action!r}"}
        await self._send(self.ap.outbox, t)  # vs_enable / vs_disable go out now, not at the next tick
        self.ap.outbox = []
        return reply

    async def _manual(self, action: str, msg: dict, control: bool) -> dict | None:
        """Manual flight from the dashboard (Tello only: a DJI aircraft is flown from its remote).

        EMERGENCY and LAND work from every screen, like RELEASE; TAKEOFF and the sticks only from
        the control laptop and only in MANUAL.
        """
        t = time.monotonic()
        if self.cfg.link.backend != "tello":
            return {"ok": False, "message": "manual flight is for the Tello; fly a DJI aircraft with its remote"}
        if action in ("land", "emergency"):
            self.ap.operator("release", t)  # leave AUTO and clear the lock first
            await self._send(self.ap.outbox, t)
            self.ap.outbox = []
            ok = await self.link.send({"type": action})
            self.logger("manual", t, action=action, ok=ok)
            return {"ok": ok, "message": "EMERGENCY: motors stopped" if action == "emergency" else "landing"}
        if not control:
            return {"ok": False, "message": "view-only screen: flying is only from the ground-station laptop"}
        if self.ap.fsm.mode != Mode.MANUAL:
            return None if action == "stick" else {"ok": False, "message": f"{self.ap.fsm.mode.value}: RELEASE first"}
        if action == "takeoff":
            ok = await self.link.send({"type": "takeoff"})
            self.logger("manual", t, action="takeoff", ok=ok)
            return {"ok": ok, "message": "taking off" if ok else "takeoff refused (battery? already flying?)"}
        sticks = {k: max(-1.0, min(1.0, float(msg.get(k, 0.0)))) for k in ("fwd", "right", "up", "yaw")}
        await self.link.send({"type": "rc", **sticks})
        return None

    # --------------------------------------------------------------- loops

    async def _send(self, msgs: list[dict], t: float) -> None:
        for m in msgs:
            ok = await self.link.send(m)
            self._tx_count += 1
            if m.get("type") != "vel":  # velocity commands are logged by the core as "cmd"
                self.logger("tx", t, msg=m, ok=ok)

    async def control_loop(self) -> None:
        period = 1.0 / self.cfg.controller.rate_hz
        next_t = time.monotonic()
        while True:
            t = time.monotonic()
            await self._send(self.ap.tick(t), t)
            next_t += period
            delay = next_t - time.monotonic()
            if delay < -period:  # we stalled; don't try to catch up with a burst
                next_t = time.monotonic()
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))

    async def publish_loop(self) -> None:
        d = self.cfg.dashboard
        period = 1.0 / d.state_hz
        animal_every = max(1, round(d.state_hz / d.animal_hz))
        n = 0
        while True:
            t = time.monotonic()
            wall = t + self.logger.wall_offset_s
            snap = self.ap.snapshot(t)
            vs = self.video_stats
            snap.update(
                type="state",
                wall_ms=int(wall * 1000),
                video={"connected": vs.connected, "fps": round(vs.fps, 1), "w": vs.width, "h": vs.height},
                lag_s=self.cfg.video.lag_s,
                infer_ms=round(self.infer_ms, 1),
                candidates=self.ap.candidate_ids(),
                tile_search=self._tile_now(),
            )
            self.hub.publish(_clean(snap))
            if n % animal_every == 0:
                a = self.ap.animal(t)
                if a is not None:
                    a.update(type="animal", last_seen_ms=int((wall - a["age_s"]) * 1000))
                    self.hub.publish(_clean(a))
            new = list(self.ap.alerts)[-(self.ap.alert_count - self._alerts_sent):] if self.ap.alert_count > self._alerts_sent else []
            for al in new:
                self.hub.publish({"type": "alert", "level": al.level, "text": al.text,
                                  "wall_ms": int((al.t + self.logger.wall_offset_s) * 1000)})
            self._alerts_sent = self.ap.alert_count
            n += 1
            await asyncio.sleep(period)

    def export(self, kind: str) -> tuple[str, str]:
        name = f"wildtrack-{self.logger.dir.name}"
        fixes = self.ap.sortie_fixes
        if kind == "gpx":
            return to_gpx(fixes, self.logger.wall_offset_s, name), name + ".gpx"
        return to_kml(fixes, self.logger.wall_offset_s, name), name + ".kml"

    async def run(self, extra: list[Awaitable[Any]] | None = None) -> None:
        import uvicorn

        self.loop = asyncio.get_running_loop()
        app = create_app(self.hub, self.tiles, self.export)
        server = uvicorn.Server(uvicorn.Config(app, host=self.cfg.dashboard.host, port=self.cfg.dashboard.port,
                                               log_level="warning", ws="websockets-sansio"))
        self.worker.start()
        log.info("dashboard: http://localhost:%d  (phones: http://<laptop-ip>:%d)", self.cfg.dashboard.port, self.cfg.dashboard.port)
        tasks = [asyncio.create_task(c) for c in (self.link.run(), self.control_loop(), self.publish_loop(), server.serve(), *(extra or []))]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    log.error("task failed", exc_info=task.exception())
        finally:
            await self.shutdown()
            for task in tasks:
                task.cancel()

    async def shutdown(self) -> None:
        if self.ap.fsm.mode in (Mode.AUTO, Mode.LOST):
            log.warning("shutting down while flying: handing back to the pilot")
            try:
                await asyncio.wait_for(self.link.send({"type": "vs_disable"}), 1.0)
            except Exception:
                log.error("could not send vs_disable; the phone watchdog hands back in 3 s")
        self.worker.stop()
        if self.ap.sortie_fixes:
            for kind in ("gpx", "kml"):
                text, _ = self.export(kind)
                self.logger.write_file(f"animal.{kind}", text)
        self.logger.close()


def _clean(x: Any) -> Any:
    """JSON can't carry NaN/inf: turn them into null."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x
