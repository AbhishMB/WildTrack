"""Autopilot core: lock rules + geolocation + estimator + controller + mode machine.

No I/O: inputs are telemetry, phone events, detections and operator actions, each
stamped with laptop monotonic time; output is a list of wire messages for the phone.
The live app (app.py), the simulators (sim/) and replay.py all drive this one class,
so the code that flies is the code that is tested.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from .config import Config
from .controller import DroneState, FollowController
from .estimator import CVKalman
from .fsm import Act, Ev, Mode, ModeMachine, Outcome, World, guard_engage
from .geo import TerrainFn, box_point, focal_px, geolocate, ground_height, latlon_to_ned, ned_to_latlon
from .link import PoseRing
from .lock import LockManager
from .models import Command, Detection, Fix, LatLon, Pose, Telemetry

LogFn = Callable[..., None]  # log(kind, t, **fields)

# Reasons DJI gives for ending Virtual Stick, as shown to the pilot.
VS_EXIT_TEXT = {
    "RC_PAUSE_STOP": "Pause button pressed",
    "RC_NOT_P_MODE": "flight-mode switch moved off P",
    "RC_LOST": "remote-controller signal lost",
    "RC_ONE_KEY_GO_HOME": "return-to-home pressed",
    "BATTERY_LOW_GO_HOME": "low-battery return-to-home",
    "BATTERY_SUPER_LOW_LANDING": "critical-battery landing",
    "NEAR_BOUNDARY": "near a restricted zone",
}


@dataclass
class Alert:
    t: float
    level: str  # info | warn | alarm
    text: str


class Autopilot:
    def __init__(self, cfg: Config, terrain: TerrainFn | None, log: LogFn | None = None):
        self.cfg = cfg
        self.terrain = terrain
        self.log: LogFn = log or (lambda kind, t, **kw: None)
        self.fsm = ModeMachine(cfg.safety)
        self.locks = LockManager(cfg.lock, cfg.classes)
        self.kf = CVKalman(cfg.estimator)
        self.ctrl = FollowController(cfg.controller, cfg.safety)
        self.poses = PoseRing(cfg.link.telemetry_buffer_s)

        self.tel: Telemetry | None = None
        self.tel_t = -math.inf
        self.home: LatLon | None = None
        self.z_home = 0.0

        self.dets: list[Detection] = []
        self.frame_size = (0, 0)
        self.target_det: Detection | None = None
        self.box_offset_deg: float | None = None
        self.last_det_rx: float | None = None  # when we last saw the target (laptop time)
        self.last_fix_rx: float | None = None  # when we last geolocated it
        self.last_fix: Fix | None = None
        self.published: dict[str, Any] | None = None  # last confirmed position for officials
        self.trail: deque[Fix] = deque()
        self.sortie_fixes: list[Fix] = []

        self.alerts: deque[Alert] = deque(maxlen=100)
        self.alert_count = 0  # total raised, so the dashboard can send only new ones
        self.outbox: list[dict] = []
        self.vel_seq = 0
        self.engaged_t: float | None = None
        self.last_gimbal_t = -math.inf
        self.hold_gimbal_pitch: float | None = None
        self.cmd_times: deque[float] = deque()
        self.last_cmd: Command | None = None
        self._last_refusals: tuple[str, ...] = ()

    # ------------------------------------------------------------------ inputs

    def on_telemetry(self, tel: Telemetry, t: float) -> None:
        self.tel, self.tel_t = tel, t
        if tel.position_valid:
            self.poses.push(t, tel)
        if tel.home is not None:
            if self.home is None:
                self.home = tel.home
                self.z_home = ground_height(tel.home.lat, tel.home.lon, 0.0, self.terrain)
                self.log("home", t, lat=tel.home.lat, lon=tel.home.lon, z_home=self.z_home)
            else:
                dn, de = latlon_to_ned(tel.home, self.home, self.cfg.geo.earth_radius_m)
                if math.hypot(dn, de) > 1.0:
                    # The local frame must not jump mid-sortie; keep the first home point.
                    self.log("home_moved", t, lat=tel.home.lat, lon=tel.home.lon)

    def on_link(self, up: bool, t: float) -> None:
        self.log("link", t, up=up)
        if not up:
            self._alert(t, "warn", "phone link down")

    def on_phone_event(self, name: str, reason: str, t: float) -> None:
        self.log("phone_event", t, name=name, reason=reason)
        if name in ("vs_exit", "vs_enable_failed"):
            text = VS_EXIT_TEXT.get(reason, reason or "unknown reason")
            if self.fsm.mode in (Mode.AUTO, Mode.LOST):
                self._alert(t, "alarm", f"Virtual Stick ended by DJI: {text}. Pilot has control.")
            self._event(Ev.VS_EXIT, t)
        elif name == "watchdog_handback":
            self._alert(t, "alarm", "phone watchdog: no commands for 3 s, Virtual Stick off. Pilot has control.")
            self._event(Ev.LAPTOP_SILENT, t)
        elif name == "watchdog_hover":
            self._alert(t, "warn", "phone watchdog: no commands for 300 ms, hovering")
        elif name == "low_battery":
            self._alert(t, "warn", f"aircraft low battery {reason}")
        elif name == "vs_enabled":
            self.log("vs_enabled", t)

    def on_detections(self, dets: list[Detection], width: int, height: int, t: float) -> None:
        """One processed frame. dets[*].frame_t is the capture time; t is now."""
        self.dets = dets
        self.frame_size = (width, height)
        frame_t = dets[0].frame_t if dets else t - self.cfg.video.lag_s
        self.locks.observe(dets, frame_t)
        mode = self.fsm.mode

        if mode == Mode.MANUAL:
            if self.cfg.lock.auto_lock:
                d = self.locks.auto_lock_ready(dets)
                if d is not None:
                    self._event(Ev.AUTO_LOCK, t, det=d)
            return

        pose = self.poses.at(frame_t, self.cfg.link.pose_max_extrapolation_s)
        det = self.locks.find_target(dets)
        if det is None and pose is not None and self.kf.initialized:
            det = self.locks.reacquire(dets, lambda d: self._gate_distance(d, pose))
            if det is not None:
                self.log("reacquire", t, track_id=det.track_id, cls=det.cls)
        if det is None:
            self.target_det = None
            return

        fix = self._geolocate(det, pose) if pose is not None else None
        if mode == Mode.LOST:
            # Resume only on a fix of the same class inside the Kalman gate.
            if fix is None or (self.kf.initialized and self.kf.mahalanobis(fix) >= self.cfg.lock.gate_mahalanobis):
                return
        self.target_det = det
        self.last_det_rx = t
        f = focal_px(width, self.cfg.camera.hfov_deg)
        self.box_offset_deg = math.degrees(math.atan2(0.5 * (det.box_xyxy[1] + det.box_xyxy[3]) - height / 2.0, f))
        if fix is None:
            return
        if not self.kf.update(fix):
            return
        self._record_fix(fix, t)
        if mode == Mode.LOST:
            self._event(Ev.REACQUIRED, t)

    def operator(self, action: str, t: float, track_id: int | None = None) -> dict:
        """LOCK (a track, or the best candidate), ENGAGE, RELEASE. Returns {ok, message}."""
        self.log("operator", t, action=action, track_id=track_id)
        if action == "release":
            # RELEASE always works and always turns Virtual Stick off, whatever the mode.
            self.outbox.append({"type": "vs_disable"})
            out = self._event(Ev.RELEASE, t)
            return {"ok": True, "message": f"released ({out.dst.value})"}
        if action == "lock":
            if self.fsm.mode != Mode.MANUAL:
                return {"ok": False, "message": f"already {self.fsm.mode.value}; RELEASE first"}
            if track_id is None:
                det = self.locks.best_candidate(self.dets)
                if det is None:
                    return {"ok": False, "message": "no candidate (a target held 5 frames at >= 0.5)"}
            else:
                det = next((d for d in self.dets if d.track_id == track_id), None)
                if det is None:
                    return {"ok": False, "message": f"track {track_id} not in the current frame"}
            self.log("lock_request", t, track_id=det.track_id, cls=det.cls, box=list(det.box_xyxy), frame_t=det.frame_t)
            out = self._event(Ev.LOCK, t, det=det)
            return {"ok": out.accepted, "message": self._describe(out)}
        if action == "engage":
            out = self._event(Ev.ENGAGE, t)
            return {"ok": out.accepted, "message": self._describe(out)}
        return {"ok": False, "message": f"unknown action {action!r}"}

    # ------------------------------------------------------------------ 10 Hz

    def tick(self, t: float) -> list[dict]:
        """Run timers and safety checks; return the messages to send to the phone now."""
        mode = self.fsm.mode
        link_alive = self.link_alive(t)

        if mode == Mode.LOCKED and self.last_det_rx is not None and t - self.last_det_rx > self.cfg.lock.track_lost_s:
            self._alert(t, "info", "target lost while locked: back to MANUAL")
            self._event(Ev.TRACK_LOST, t)
        elif mode == Mode.AUTO and self._since_fix(t) > self.cfg.safety.no_fix_hover_s:
            self._alert(t, "warn", "no fix for %.0f s: hovering" % self.cfg.safety.no_fix_hover_s)
            self._event(Ev.NO_FIX, t)
        elif mode == Mode.LOST and self._since_fix(t) > self.cfg.safety.no_fix_handback_s:
            self._alert(t, "alarm", "animal lost for %.0f s: Virtual Stick off. Pilot has control." % self.cfg.safety.no_fix_handback_s)
            self._event(Ev.LOST_TIMEOUT, t)

        mode = self.fsm.mode
        if mode in (Mode.AUTO, Mode.LOST):
            self._safety_checks(t, link_alive)

        if self.fsm.mode == Mode.LOCKED and self.cfg.lock.auto_engage:  # simulation only
            refusals = guard_engage(self.world(t), self.cfg.safety)
            if refusals:
                self._note_refusals(t, refusals)
            else:
                self._event(Ev.ENGAGE, t)

        msgs = self._commands(t)
        out_msgs, self.outbox = self.outbox + msgs, []
        return out_msgs

    def _note_refusals(self, t: float, refusals: list[str]) -> None:
        if tuple(refusals) != self._last_refusals:
            self._last_refusals = tuple(refusals)
            self.log("auto_engage_wait", t, refusals=refusals)

    def _safety_checks(self, t: float, link_alive: bool) -> None:
        s = self.cfg.safety
        tel = self.tel
        if not link_alive:
            self._alert(t, "alarm", "phone link lost in AUTO: phone watchdog hovers, then hands back")
            self._event(Ev.LINK_LOST, t)
            return
        assert tel is not None
        if tel.battery_pct < s.min_battery_pct:
            self._alert(t, "alarm", f"battery {tel.battery_pct:.0f}% < {s.min_battery_pct:.0f}%: pilot has control")
            self._event(Ev.LOW_BATTERY, t)
            return
        d = self.dist_home()
        if d is not None and d > s.operating_radius_m:
            self._alert(t, "alarm", f"{d:.0f} m from home, beyond {s.operating_radius_m:.0f} m: pilot has control")
            self._event(Ev.OUT_OF_RADIUS, t)
            return
        if self.engaged_t is not None and t - self.engaged_t > s.vs_grace_s and not tel.vs_enabled:
            self._alert(t, "alarm", "telemetry shows Virtual Stick off: pilot has control")
            self._event(Ev.VS_EXIT, t)

    def _commands(self, t: float) -> list[dict]:
        mode = self.fsm.mode
        if mode not in (Mode.AUTO, Mode.LOST) or self.tel is None:
            return []
        drone = self.drone_state(t)
        if drone is None:
            return []
        msgs: list[dict] = []
        if mode == Mode.AUTO and self.kf.initialized:
            est = self.kf.peek(t)
            cmd = self.ctrl.step(drone, est, self.box_offset_deg)
            self.box_offset_deg = None  # use each box correction once
            self.hold_gimbal_pitch = cmd.gimbal_pitch
        else:
            hold = self.hold_gimbal_pitch if self.hold_gimbal_pitch is not None else self.tel.gimbal.pitch
            cmd = self.ctrl.hover(t, hold)
        self.vel_seq += 1
        msgs.append(
            {
                "type": "vel",
                "seq": self.vel_seq,
                "vn": round(cmd.vn, 3),
                "ve": round(cmd.ve, 3),
                "vd": round(cmd.vd, 3),
                "yaw_rate_dps": round(cmd.yaw_rate, 2),
            }
        )
        if mode == Mode.AUTO and t - self.last_gimbal_t >= 1.0 / self.cfg.controller.gimbal_rate_hz:
            self.last_gimbal_t = t
            msgs.append({"type": "gimbal", "pitch_deg": round(cmd.gimbal_pitch, 1)})
        self.last_cmd = cmd
        self.cmd_times.append(t)
        while self.cmd_times and self.cmd_times[0] < t - 2.0:
            self.cmd_times.popleft()
        self.log("cmd", t, vn=cmd.vn, ve=cmd.ve, vd=cmd.vd, yaw_rate=cmd.yaw_rate, gimbal=cmd.gimbal_pitch, mode=mode.value)
        return msgs

    # ------------------------------------------------------------------ events and actions

    def _event(self, ev: Ev, t: float, det: Detection | None = None) -> Outcome:
        w = self.world(t, det)
        out = self.fsm.handle(ev, w)
        if out.accepted:
            self._apply(out, t, det)
        else:
            self.log("refused", t, event=ev.value, mode=out.src.value, refusals=list(out.refusals))
        return out

    def _apply(self, out: Outcome, t: float, det: Detection | None = None) -> None:
        self.log("mode", t, event=out.event.value, src=out.src.value, dst=out.dst.value, actions=[a.value for a in out.actions])
        for a in out.actions:
            if a == Act.FIX_TARGET:
                assert det is not None
                self.locks.lock(det)
                self.kf.reset()
                self.trail.clear()
                self.target_det = det
                self.last_det_rx = t
                self.last_fix_rx = None
                pose = self.poses.at(det.frame_t, self.cfg.link.pose_max_extrapolation_s)
                fix = self._geolocate(det, pose) if pose is not None else None
                if fix is not None:
                    self.kf.init(fix)
                    self._record_fix(fix, t)
                self._alert(t, "info", f"locked {det.cls} #{det.track_id}")
            elif a == Act.CLEAR_TARGET:
                self.locks.release()
                self.kf.reset()
                self.target_det = None
                self.engaged_t = None
                self.hold_gimbal_pitch = None
            elif a == Act.VS_ENABLE:
                self.outbox.append({"type": "vs_enable"})
                self.engaged_t = t
                self._alert(t, "info", "ENGAGED: laptop is flying")
            elif a == Act.RESET_INTEGRATOR:
                vn, ve = (self.tel.vel_ned[0], self.tel.vel_ned[1]) if self.tel else (0.0, 0.0)
                self.ctrl.reset(vn if math.isfinite(vn) else 0.0, ve if math.isfinite(ve) else 0.0)
            elif a == Act.VS_DISABLE:
                if not any(m.get("type") == "vs_disable" for m in self.outbox):
                    self.outbox.append({"type": "vs_disable"})
            elif a == Act.HOVER:
                pass  # _commands sends zero velocity every tick while LOST
            elif a == Act.ALERT_PILOT:
                pass  # the alert text was raised by whoever detected the condition
            # START_COMMANDS / STOP_COMMANDS: commands are sent exactly while the mode is AUTO or LOST

    def _describe(self, out: Outcome) -> str:
        if out.accepted:
            return f"{out.src.value} -> {out.dst.value}"
        return "refused: " + "; ".join(out.refusals)

    def _alert(self, t: float, level: str, text: str) -> None:
        self.alerts.append(Alert(t, level, text))
        self.alert_count += 1
        self.log("alert", t, level=level, text=text)

    # ------------------------------------------------------------------ geometry helpers

    def _geolocate(self, det: Detection, pose: Pose) -> Fix | None:
        if self.home is None:
            return None
        w, h = self.frame_size
        u, v = box_point(det.box_xyxy, pose.gimbal_att.pitch, self.cfg.geo.nadir_pitch_deg)
        return geolocate(pose, u, v, w, h, self.cfg.camera.hfov_deg, self.home, self.z_home, self.terrain, self.cfg.geo)

    def _gate_distance(self, det: Detection, pose: Pose) -> float | None:
        fix = self._geolocate(det, pose)
        return None if fix is None else self.kf.mahalanobis(fix)

    def _record_fix(self, fix: Fix, t: float) -> None:
        self.last_fix = fix
        self.last_fix_rx = t
        self.trail.append(fix)
        self.sortie_fixes.append(fix)
        horizon = self.cfg.dashboard.trail_minutes * 60.0
        while self.trail and self.trail[0].t < fix.t - horizon:
            self.trail.popleft()
        est = self.kf.estimate()
        assert self.home is not None
        p = ned_to_latlon(est.n, est.e, self.home, self.cfg.geo.earth_radius_m)
        # Position: filtered estimate at the sighting. Radius: the single-fix 1-sigma, because
        # attitude and height errors are biases that do not average out over consecutive fixes.
        self.published = {
            "lat": p.lat,
            "lon": p.lon,
            "sigma_m": fix.sigma_m,
            "t": fix.t,
            "speed_mps": est.speed,
            "heading_deg": est.heading_deg,
            "cls": self.locks.locked_cls,
            "track_id": self.locks.locked_id,
        }
        self.log("fix", t, frame_t=fix.t, lat=fix.lat, lon=fix.lon, n=fix.n, e=fix.e, sigma_m=fix.sigma_m,
                 est_lat=p.lat, est_lon=p.lon, vn=est.vn, ve=est.ve)

    def _since_fix(self, t: float) -> float:
        return math.inf if self.last_fix_rx is None else t - self.last_fix_rx

    def link_alive(self, t: float) -> bool:
        return t - self.tel_t <= self.cfg.link.alive_s

    def drone_ne(self) -> tuple[float, float] | None:
        if self.tel is None or self.home is None or not self.tel.position_valid:
            return None
        return latlon_to_ned(LatLon(self.tel.lat, self.tel.lon), self.home, self.cfg.geo.earth_radius_m)

    def dist_home(self) -> float | None:
        ne = self.drone_ne()
        return None if ne is None else math.hypot(*ne)

    def agl(self) -> float | None:
        if self.tel is None or not self.tel.position_valid:
            return None
        return self.z_home + self.tel.alt_rel_m - ground_height(self.tel.lat, self.tel.lon, self.z_home, self.terrain)

    def drone_state(self, t: float) -> DroneState | None:
        ne, agl, tel = self.drone_ne(), self.agl(), self.tel
        if ne is None or agl is None or tel is None or self.home is None:
            return None
        h_target = agl
        if self.kf.initialized:
            est = self.kf.peek(t)
            p = ned_to_latlon(est.n, est.e, self.home, self.cfg.geo.earth_radius_m)
            h_target = self.z_home + tel.alt_rel_m - ground_height(p.lat, p.lon, self.z_home, self.terrain)
        vn, ve = tel.vel_ned[0], tel.vel_ned[1]
        return DroneState(t, ne[0], ne[1], vn if math.isfinite(vn) else 0.0, ve if math.isfinite(ve) else 0.0,
                          tel.att.yaw, agl, h_target)

    def world(self, t: float, det: Detection | None = None) -> World:
        tel = self.tel
        return World(
            target_lockable=det is not None and self.locks.is_lockable(det.cls),
            auto_lock_enabled=self.cfg.lock.auto_lock,
            gps_sats=tel.gps_sats if tel else 0,
            battery_pct=tel.battery_pct if tel and math.isfinite(tel.battery_pct) else 0.0,
            agl_m=self.agl(),
            dist_home_m=self.dist_home(),
            link_alive=self.link_alive(t),
            has_fix=self.kf.initialized,
        )

    # ------------------------------------------------------------------ dashboard

    def snapshot(self, t: float) -> dict[str, Any]:
        tel = self.tel
        w = self.world(t)
        refusals = guard_engage(w, self.cfg.safety)
        if self.fsm.mode != Mode.LOCKED:
            refusals = [f"mode is {self.fsm.mode.value}; ENGAGE needs LOCKED"] + refusals
        est = None
        if self.kf.initialized and self.home is not None:
            e = self.kf.peek(t)
            p = ned_to_latlon(e.n, e.e, self.home, self.cfg.geo.earth_radius_m)
            est = {"lat": p.lat, "lon": p.lon, "sigma_m": e.sigma_m, "speed_mps": e.speed, "heading_deg": e.heading_deg}
        rate = (len(self.cmd_times) - 1) / (self.cmd_times[-1] - self.cmd_times[0]) if len(self.cmd_times) > 2 else 0.0
        return {
            "mode": self.fsm.mode.value,
            "link_alive": w.link_alive,
            "drone": None
            if tel is None
            else {
                "lat": tel.lat if tel.position_valid else None,
                "lon": tel.lon if tel.position_valid else None,
                "alt_rel_m": tel.alt_rel_m,
                "agl_m": self.agl(),
                "yaw_deg": tel.att.yaw,
                "gimbal_pitch_deg": tel.gimbal.pitch,
                "speed_mps": math.hypot(tel.vel_ned[0], tel.vel_ned[1]) if math.isfinite(tel.vel_ned[0]) else None,
                "battery_pct": tel.battery_pct,
                "gps_sats": tel.gps_sats,
                "vs_enabled": tel.vs_enabled,
                "flight_mode": tel.flight_mode,
                "dist_home_m": self.dist_home(),
            },
            "home": None if self.home is None else {"lat": self.home.lat, "lon": self.home.lon},
            "target": None
            if self.locks.locked_id is None
            else {"track_id": self.locks.locked_id, "cls": self.locks.locked_cls, "visible": self.target_det is not None},
            "estimate": est,
            "engage_refusals": refusals,
            "cmd_rate_hz": rate,
            "last_cmd": None
            if self.last_cmd is None
            else {"vn": self.last_cmd.vn, "ve": self.last_cmd.ve, "vd": self.last_cmd.vd,
                  "yaw_rate": self.last_cmd.yaw_rate, "gimbal_pitch": self.last_cmd.gimbal_pitch},
        }

    def animal(self, t: float) -> dict[str, Any] | None:
        """The published position: what officials see, once a second while locked."""
        if self.published is None:
            return None
        p = dict(self.published)
        p["age_s"] = t - p.pop("t")
        p["locked"] = self.locks.locked
        p["trail"] = [[f.lat, f.lon] for f in self.trail]
        return p

    def candidate_ids(self) -> list[int]:
        return [d.track_id for d in self.locks.candidates(self.dets)]
