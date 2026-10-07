"""Kinematic aircraft (first-order velocity response plus command delay) and a stand-in
for the Android bridge app: telemetry out, commands in, Virtual Stick and the watchdog.

SimPhone mirrors android-bridge/VirtualStick.kt and Watchdog.kt, so the laptop can be
tested end to end, including hover after 300 ms and hand-back after 3 s of silence.
"""

from __future__ import annotations

import math
from collections import deque

from ..config import SimCfg
from ..geo import ned_to_latlon
from ..models import LatLon
from ..watchdog import Watchdog  # noqa: F401  (re-exported for the tests)


class SimDrone:
    def __init__(self, cfg: SimCfg, n: float, e: float, h: float, yaw_deg: float, gimbal_pitch: float):
        self.cfg = cfg
        self.n, self.e, self.h = n, e, h  # h: metres above home
        self.yaw = yaw_deg
        self.vn = self.ve = self.vd = self.yaw_rate = 0.0
        self.cmd = (0.0, 0.0, 0.0, 0.0)
        self.pending: deque[tuple[float, tuple[float, float, float, float]]] = deque()
        self.vs_enabled = False
        self.gimbal_pitch = gimbal_pitch
        self.gimbal_target = gimbal_pitch
        self.battery = cfg.battery_start_pct

    def command(self, t: float, vn: float, ve: float, vd: float, yaw_rate: float) -> None:
        self.pending.append((t + self.cfg.cmd_delay_s, (vn, ve, vd, yaw_rate)))

    def step(self, t: float, dt: float) -> None:
        while self.pending and self.pending[0][0] <= t:
            c = self.pending.popleft()[1]
            if self.vs_enabled:
                self.cmd = c
        if not self.vs_enabled:
            self.cmd = (0.0, 0.0, 0.0, 0.0)  # the pilot holds the sticks centred: hover
            self.pending.clear()
        k = min(1.0, dt / self.cfg.vel_tau_s)
        self.vn += (self.cmd[0] - self.vn) * k
        self.ve += (self.cmd[1] - self.ve) * k
        self.vd += (self.cmd[2] - self.vd) * k
        self.yaw_rate += (self.cmd[3] - self.yaw_rate) * k
        self.n += self.vn * dt
        self.e += self.ve * dt
        self.h -= self.vd * dt
        self.yaw = (self.yaw + self.yaw_rate * dt + 180.0) % 360.0 - 180.0
        step = self.cfg.gimbal_rate_dps * dt
        self.gimbal_pitch += max(-step, min(step, self.gimbal_target - self.gimbal_pitch))
        self.battery -= self.cfg.battery_drain_pct_per_min * dt / 60.0


class SimPhone:
    """The bridge app's behaviour. Returns phone -> laptop event messages."""

    def __init__(self, cfg: SimCfg, drone: SimDrone, home: LatLon, earth_radius_m: float):
        self.cfg = cfg
        self.drone = drone
        self.home = home
        self.R = earth_radius_m
        self.watchdog = Watchdog(cfg.watchdog_hover_s, cfg.watchdog_handback_s)
        self.seq = 0

    @staticmethod
    def _event(name: str, reason: str = "") -> dict:
        return {"type": "event", "name": name, "reason": reason}

    def handle(self, msg: dict, t: float) -> list[dict]:
        kind = msg.get("type")
        d = self.drone
        if kind == "vel":
            if d.vs_enabled:
                self.watchdog.feed(t)
                d.command(t, float(msg["vn"]), float(msg["ve"]), float(msg["vd"]), float(msg["yaw_rate_dps"]))
        elif kind == "gimbal":
            d.gimbal_target = max(-90.0, min(0.0, float(msg["pitch_deg"])))
        elif kind == "vs_enable":
            d.vs_enabled = True
            self.watchdog.feed(t)
            return [self._event("vs_enabled")]
        elif kind == "vs_disable":
            if d.vs_enabled:
                d.vs_enabled = False
                self.watchdog.disarm()
                return [self._event("vs_exit", "MSDK_REQUEST")]
        return []

    def check(self, t: float) -> list[dict]:
        if not self.drone.vs_enabled:
            return []
        w = self.watchdog.check(t)
        if w == Watchdog.HOVER:
            self.drone.command(t - self.cfg.cmd_delay_s, 0.0, 0.0, 0.0, 0.0)  # immediate, phone-side
            return [self._event("watchdog_hover")]
        if w == Watchdog.HANDBACK:
            self.drone.vs_enabled = False
            return [self._event("watchdog_handback"), self._event("vs_exit", "WATCHDOG")]
        return []

    def pause(self) -> list[dict]:
        """The pilot presses Pause on the remote: DJI ends Virtual Stick."""
        if not self.drone.vs_enabled:
            return []
        self.drone.vs_enabled = False
        self.watchdog.disarm()
        return [self._event("vs_exit", "RC_PAUSE_STOP")]

    def telemetry(self) -> dict:
        d = self.drone
        self.seq += 1
        p = ned_to_latlon(d.n, d.e, self.home, self.R)
        return {
            "type": "telemetry",
            "seq": self.seq,
            "t_phone_ms": 0,
            "lat": p.lat,
            "lon": p.lon,
            "alt_rel_m": d.h,
            "att": {"pitch": 0.0, "roll": 0.0, "yaw": d.yaw},
            "vel_ned": {"n": d.vn, "e": d.ve, "d": d.vd},
            "gimbal": {"pitch": d.gimbal_pitch, "roll": 0.0, "yaw": d.yaw},
            "battery_pct": round(d.battery, 1),
            "gps_sats": self.cfg.gps_sats,
            "vs_enabled": d.vs_enabled,
            "flight_mode": "P",
            "home": {"lat": self.home.lat, "lon": self.home.lon},
        }


def start_drone(cfg: SimCfg, target_n: float, target_e: float) -> SimDrone:
    """Where the pilot has flown to: standoff from the animal, nose toward it, gimbal down."""
    ang = math.radians(225.0)
    n = target_n + cfg.start_standoff_m * math.cos(ang)
    e = target_e + cfg.start_standoff_m * math.sin(ang)
    yaw = math.degrees(math.atan2(target_e - e, target_n - n))
    return SimDrone(cfg, n, e, cfg.start_alt_m, yaw, cfg.start_gimbal_deg)
