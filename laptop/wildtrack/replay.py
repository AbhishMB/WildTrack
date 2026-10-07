"""Rerun detection, geolocation, the estimator and the mode logic on a logged sortie.

Feeds the logged telemetry, phone events and operator actions to the current code at
their logged times. Detection is either rerun on video.mp4 with the current weights,
or taken from the log (--logged-detections). Commands are computed but go nowhere.
Prints how the new run differs from what happened in flight, so a code change can be
checked against real data before the next flight.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config, override
from .core import Autopilot
from .dem import Terrain
from .export import to_gpx, to_kml
from .logger import read_events
from .models import Detection, Telemetry
from .video_in import Frame

log = logging.getLogger(__name__)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class VideoFrames:
    """Sequential reader of video.mp4 by frame index."""

    def __init__(self, path: Path):
        import cv2

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.next_idx = 0

    def get(self, idx: int) -> np.ndarray | None:
        img = None
        while self.next_idx <= idx:
            ok, img = self.cap.read()
            if not ok:
                return None
            self.next_idx += 1
        return img


@dataclass
class ReplayResult:
    fixes: list = field(default_factory=list)
    transitions: list = field(default_factory=list)
    logged_fixes: list = field(default_factory=list)
    logged_transitions: list = field(default_factory=list)
    cmd_diff_mps: list = field(default_factory=list)


def replay(cfg: Config, log_dir: Path, rerun_detector=None, tick_hz: float | None = None) -> tuple[ReplayResult, Autopilot]:
    log_dir = Path(log_dir)
    meta = json.loads((log_dir / "meta.json").read_text(encoding="utf-8"))
    logged_lag = meta["config"]["video"]["lag_s"]
    if abs(cfg.video.lag_s - logged_lag) > 1e-9:
        log.info("using the sortie's video lag %.3f s (current config says %.3f s)", logged_lag, cfg.video.lag_s)
        cfg = override(cfg, "video.lag_s", logged_lag)
    events = read_events(log_dir)
    terrain = Terrain.from_dir(cfg.path(cfg.geo.dem_dir))
    res = ReplayResult()

    def on_log(kind, t, **kw):
        if kind == "fix":
            res.fixes.append(kw | {"t": t})
        elif kind == "mode":
            res.transitions.append((t, kw["src"], kw["dst"], kw["event"]))

    ap = Autopilot(cfg, terrain, log=on_log)
    video = VideoFrames(log_dir / "video.mp4") if rerun_detector is not None else None
    logged_cmds = [(e["t"], e["vn"], e["ve"]) for e in events if e["kind"] == "cmd"]
    res.logged_fixes = [e for e in events if e["kind"] == "fix"]
    res.logged_transitions = [(e["t"], e["src"], e["dst"], e["event"]) for e in events if e["kind"] == "mode"]

    if not events:
        return res, ap
    period = 1.0 / (tick_hz or cfg.controller.rate_hz)
    next_tick = events[0]["t"]
    replay_cmds: list[tuple[float, float, float]] = []

    for e in events:
        t = e["t"]
        while next_tick <= t:
            for m in ap.tick(next_tick):
                if m["type"] == "vel":
                    replay_cmds.append((next_tick, m["vn"], m["ve"]))
            next_tick += period
        kind = e["kind"]
        if kind == "telemetry":
            ap.on_telemetry(Telemetry.from_msg(e["msg"]), t)
        elif kind == "phone_event":
            ap.on_phone_event(e["name"], e.get("reason", ""), t)
        elif kind == "link":
            ap.on_link(e["up"], t)
        elif kind == "operator" and e["action"] in ("release", "engage"):
            ap.operator(e["action"], t)
        elif kind == "lock_request":
            # Track IDs differ after re-running the detector: lock the box that overlaps the logged one.
            best = max((d for d in ap.dets if d.cls == e["cls"]), key=lambda d: iou(d.box_xyxy, e["box"]), default=None)
            ap.operator("lock", t, best.track_id if best is not None else None)
        elif kind == "dets":
            w, h = e["w"], e["h"]
            if rerun_detector is not None and video is not None and e.get("video_idx") is not None:
                img = video.get(e["video_idx"])
                if img is None:
                    continue
                dets = rerun_detector.detect(Frame(e["video_idx"], t, img), e["frame_t"], False)
            else:
                dets = [Detection(e["frame_t"], int(tid), cls, conf, tuple(box)) for tid, cls, conf, box in e["dets"]]
            ap.on_detections(dets, w, h, t)

    # how different are the commands the new code would have sent?
    if logged_cmds and replay_cmds:
        lt = np.array([c[0] for c in logged_cmds])
        for t, vn, ve in replay_cmds:
            k = int(np.argmin(np.abs(lt - t)))
            if abs(lt[k] - t) < period:
                res.cmd_diff_mps.append(math.hypot(vn - logged_cmds[k][1], ve - logged_cmds[k][2]))
    return res, ap


def fix_distance_stats(new: list[dict], old: list[dict]) -> tuple[int, float] | None:
    """Mean distance between replayed and logged fixes of the same frame."""
    by_frame = {round(f["frame_t"], 4): f for f in old}
    d = []
    for f in new:
        g = by_frame.get(round(f["frame_t"], 4))
        if g is not None:
            d.append(math.hypot(f["n"] - g["n"], f["e"] - g["e"]))
    return (len(d), float(np.mean(d))) if d else None


def main(cfg: Config, args) -> int:
    log_dir = Path(args.log_dir)
    detector = None
    if not args.logged_detections:
        from .detect import YoloDetector

        weights = Path(args.weights) if args.weights else cfg.path(cfg.detector.weights)
        detector = YoloDetector(cfg.detector, weights)
    res, ap = replay(cfg, log_dir, detector)
    out = Path(args.out) if args.out else log_dir / "replay"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fixes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "frame_t", "lat", "lon", "n", "e", "sigma_m"])
        for x in res.fixes:
            w.writerow([f"{x['t']:.3f}", f"{x['frame_t']:.3f}", f"{x['lat']:.7f}", f"{x['lon']:.7f}",
                        f"{x['n']:.2f}", f"{x['e']:.2f}", f"{x['sigma_m']:.2f}"])
    meta = json.loads((log_dir / "meta.json").read_text(encoding="utf-8"))
    (out / "animal.gpx").write_text(to_gpx(ap.sortie_fixes, meta["wall_offset_s"], log_dir.name + "-replay"), encoding="utf-8")
    (out / "animal.kml").write_text(to_kml(ap.sortie_fixes, meta["wall_offset_s"], log_dir.name + "-replay"), encoding="utf-8")

    print(f"replayed {log_dir}")
    print(f"  fixes:       logged {len(res.logged_fixes):5d}   replay {len(res.fixes):5d}")
    if res.fixes:
        print(f"  mean sigma:  logged {np.mean([f['sigma_m'] for f in res.logged_fixes]) if res.logged_fixes else float('nan'):5.2f} m "
              f"replay {np.mean([f['sigma_m'] for f in res.fixes]):5.2f} m")
    st = fix_distance_stats(res.fixes, res.logged_fixes)
    if st:
        print(f"  same-frame fixes: {st[0]}, mean distance between runs {st[1]:.2f} m")
    if res.cmd_diff_mps:
        print(f"  velocity commands: mean difference {np.mean(res.cmd_diff_mps):.2f} m/s, max {np.max(res.cmd_diff_mps):.2f} m/s")
    print("  mode changes (logged):")
    for t, a, b, ev in res.logged_transitions:
        print(f"    {t:10.2f}  {a:>6} -> {b:<6} ({ev})")
    print("  mode changes (replay):")
    for t, a, b, ev in res.transitions:
        print(f"    {t:10.2f}  {a:>6} -> {b:<6} ({ev})")
    print(f"  written to {out}")
    return 0
