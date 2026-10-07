"""Headless closed-loop simulation in virtual time (docs/SPEC.md, build stage 3).

Runs the real Autopilot (core.py) against the kinematic aircraft, the phone stand-in
and the synthetic camera's oracle detections, with every delay modelled: telemetry
latency, video lag, inference time, and the command delay to the aircraft.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config import Config, override
from ..core import Autopilot
from ..fsm import Mode
from ..geo import latlon_to_ned
from ..models import Detection, LatLon, Telemetry
from .camera import OracleTracker, SimCamera
from .drone import SimPhone, start_drone
from .world import GroundFn, make_world


@dataclass
class Scenario:
    duration_s: float = 120.0
    animal_speed_mps: float | None = None  # None: cfg.sim.animal_speed_mps
    tau_s: float | None = None  # total pure delay, split between video lag and command delay
    hide: tuple[float, float] | None = None  # target invisible between these sim times
    lock_at_s: float = 1.0
    engage_at_s: float = 3.0
    pause_at_s: float | None = None  # pilot presses Pause
    laptop_dies_at_s: float | None = None  # laptop stops sending anything
    trees: bool = False
    ground: GroundFn | None = None  # terrain under the sim, (n, e) -> metres
    dt_s: float = 0.01
    seed: int | None = None


@dataclass
class Trace:
    t: list[float] = field(default_factory=list)
    mode: list[str] = field(default_factory=list)
    drone: list[tuple[float, float, float]] = field(default_factory=list)
    animal: list[tuple[float, float]] = field(default_factory=list)
    estimate: list[tuple[float, float] | None] = field(default_factory=list)
    vs_enabled: list[bool] = field(default_factory=list)


@dataclass
class Result:
    trace: Trace
    transitions: list[tuple[float, str, str, str]]
    alerts: list[str]
    standoff_err_mean: float
    standoff_err_max: float
    estimate_err_mean: float
    auto_time_s: float
    final_mode: str
    phone_events: list[tuple[float, str, str]]
    autopilot: Autopilot

    def summary(self) -> str:
        return (
            f"AUTO for {self.auto_time_s:.0f} s; standoff error mean {self.standoff_err_mean:.2f} m, "
            f"max {self.standoff_err_max:.2f} m; estimate error mean {self.estimate_err_mean:.2f} m; "
            f"final mode {self.final_mode}"
        )


def _to_detections(sim_dets, ids: dict[int, int], frame_t: float) -> list[Detection]:
    return [Detection(frame_t, ids[s.animal_id], s.cls, s.conf, s.box) for s in sim_dets]


def run(cfg: Config, sc: Scenario, settle_s: float = 5.0) -> Result:
    s = cfg.sim
    if sc.tau_s is not None:
        half = sc.tau_s / 2.0
        cfg = override(cfg, "sim.video_lag_s", half)
        cfg = override(cfg, "sim.cmd_delay_s", half)
        s = cfg.sim
    cfg = override(cfg, "video.lag_s", s.video_lag_s)  # assume the lag was calibrated
    rng = np.random.default_rng(s.seed if sc.seed is None else sc.seed)
    home = LatLon(*s.home)
    R = cfg.geo.earth_radius_m

    world = make_world(s, rng, speed=sc.animal_speed_mps, trees=sc.trees, ground=sc.ground)
    if sc.hide:
        world.hide_windows.append((world.target.id, sc.hide[0], sc.hide[1]))
    drone = start_drone(s, world.target.n, world.target.e)
    phone = SimPhone(s, drone, home, R)
    cam = SimCamera(s, cfg.camera.hfov_deg)
    tracker = OracleTracker()

    terrain = None
    if sc.ground is not None:
        g = sc.ground

        def terrain(lat: float, lon: float) -> float:
            n, e = latlon_to_ned(LatLon(lat, lon), home, R)
            return g(n, e)

    ap = Autopilot(cfg, terrain)
    transitions: list[tuple[float, str, str, str]] = []
    base_log = ap.log

    def log(kind, t, **kw):
        if kind == "mode":
            transitions.append((t, kw["src"], kw["dst"], kw["event"]))
        base_log(kind, t, **kw)

    ap.log = log

    queue: list = []
    counter = itertools.count()

    def at(t: float, fn: Callable[[], None]) -> None:
        heapq.heappush(queue, (t, next(counter), fn))

    phone_events: list[tuple[float, str, str]] = []

    def to_laptop(t: float, events: list[dict]) -> None:
        for ev in events:
            phone_events.append((t, ev["name"], ev.get("reason", "")))
            at(t + s.telemetry_delay_s, lambda ev=ev: ap.on_phone_event(ev["name"], ev.get("reason", ""), now[0]))

    now = [0.0]
    trace = Trace()
    dt = sc.dt_s
    next_tel = 0.0
    next_frame = 0.0
    next_tick = 0.0
    laptop_alive = True
    locked_ok = engaged_ok = False
    paused = False
    tick_dt = 1.0 / cfg.controller.rate_hz
    frame_dt = 1.0 / s.fps

    steps = int(sc.duration_s / dt)
    for i in range(steps + 1):
        t = i * dt
        now[0] = t
        while queue and queue[0][0] <= t:
            _, _, fn = heapq.heappop(queue)
            fn()

        world.step(dt, rng)
        drone.step(t, dt)
        to_laptop(t, phone.check(t))

        if sc.pause_at_s is not None and not paused and t >= sc.pause_at_s:
            paused = True
            to_laptop(t, phone.pause())
        if sc.laptop_dies_at_s is not None and t >= sc.laptop_dies_at_s:
            laptop_alive = False

        if t >= next_tel:
            next_tel += 0.1
            msg = phone.telemetry()
            at(t + s.telemetry_delay_s, lambda msg=msg: ap.on_telemetry(Telemetry.from_msg(msg), now[0]))

        if t >= next_frame:
            next_frame += frame_dt
            sim_dets = cam.detect(drone, world, rng)
            ids = tracker.assign([sd.animal_id for sd in sim_dets])
            arrival = t + s.video_lag_s
            frame_t = arrival - cfg.video.lag_s
            dets = _to_detections(sim_dets, ids, frame_t)
            if laptop_alive:
                at(arrival + s.infer_s, lambda dets=dets: ap.on_detections(dets, s.width, s.height, now[0]))

        if laptop_alive and t >= next_tick:
            next_tick += tick_dt
            if not locked_ok and t >= sc.lock_at_s and ap.fsm.mode == Mode.MANUAL:
                tid = next((d.track_id for d in ap.dets if d.cls == world.target.cls), None)
                if tid is not None:
                    locked_ok = ap.operator("lock", t, tid)["ok"]
            if locked_ok and not engaged_ok and t >= sc.engage_at_s and ap.fsm.mode == Mode.LOCKED:
                engaged_ok = ap.operator("engage", t)["ok"]
            for m in ap.tick(t):
                at(t + s.telemetry_delay_s, lambda m=m: to_laptop(now[0], phone.handle(m, now[0])))

        trace.t.append(t)
        trace.mode.append(ap.fsm.mode.value)
        trace.drone.append((drone.n, drone.e, drone.h))
        trace.animal.append((world.target.n, world.target.e))
        est = None
        if ap.kf.initialized and ap.home is not None:
            e = ap.kf.peek(t)
            est = (e.n, e.e)
        trace.estimate.append(est)
        trace.vs_enabled.append(drone.vs_enabled)

    # metrics over AUTO, after a settling time from the first engagement
    t_arr = np.array(trace.t)
    auto = np.array([m == Mode.AUTO.value for m in trace.mode])
    first_auto = t_arr[auto][0] if auto.any() else math.inf
    mask = auto & (t_arr >= first_auto + settle_s)
    dr = np.array(trace.drone)
    an = np.array(trace.animal)
    standoff_err = np.abs(np.hypot(dr[:, 0] - an[:, 0], dr[:, 1] - an[:, 1]) - cfg.controller.standoff_m)
    est_err = np.array([math.hypot(e[0] - a[0], e[1] - a[1]) if e else np.nan for e, a in zip(trace.estimate, trace.animal)])
    return Result(
        trace=trace,
        transitions=transitions,
        alerts=[a.text for a in ap.alerts],
        standoff_err_mean=float(standoff_err[mask].mean()) if mask.any() else math.nan,
        standoff_err_max=float(standoff_err[mask].max()) if mask.any() else math.nan,
        estimate_err_mean=float(np.nanmean(est_err[mask])) if mask.any() else math.nan,
        auto_time_s=float(auto.sum() * dt),
        final_mode=ap.fsm.mode.value,
        phone_events=phone_events,
        autopilot=ap,
    )


def main(cfg: Config, args) -> int:
    sc = Scenario(
        duration_s=args.duration,
        animal_speed_mps=args.speed,
        tau_s=args.tau,
        hide=tuple(args.hide) if args.hide else None,
        trees=args.trees,
    )
    res = run(cfg, sc)
    for t, src, dst, ev in res.transitions:
        print(f"{t:7.2f}s  {src:>6} -> {dst:<6}  ({ev})")
    for a in res.alerts:
        print("  alert:", a)
    print(res.summary())
    if args.plot:
        _plot(res, cfg, args.plot)
        print("plot written to", args.plot)
    return 0


def _plot(res: Result, cfg: Config, path: str) -> None:
    import cv2

    dr = np.array(res.trace.drone)
    an = np.array(res.trace.animal)
    pts = np.vstack([dr[:, :2], an])
    lo, hi = pts.min(0) - 20, pts.max(0) + 20
    size = 800
    scale = size / float(max(hi - lo))
    img = np.full((size, size, 3), 255, np.uint8)

    def px(n, e):
        return int((e - lo[1]) * scale), int(size - (n - lo[0]) * scale)

    for arr, col in ((an, (0, 0, 200)), (dr[:, :2], (200, 0, 0))):
        for k in range(1, len(arr), 10):
            cv2.line(img, px(*arr[k - 10 if k >= 10 else 0]), px(*arr[k]), col, 2)
    cv2.putText(img, "red: animal  blue: drone", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, res.summary()[:90], (10, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(path, img)
