"""Follow controller (docs/SPEC.md, "Tracking controller"). Pure: no I/O.

The controller does not steer on pixels. It flies the drone to a standoff point beside
the estimated animal position and commands ground-frame velocity:

    v_cmd = v_animal + Kp * e + Ki * integral(e),   e = p_set - p_drone

then limits speed, acceleration and the integrator. Yaw turns the nose toward the
animal; gimbal pitch comes from geometry plus a small correction from the box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ControllerCfg, SafetyCfg
from .geo import bearing_deg, wrap180
from .models import Command, TargetEstimate


@dataclass(frozen=True)
class DroneState:
    t: float
    n: float  # metres from home
    e: float
    vn: float
    ve: float
    yaw_deg: float
    agl_m: float  # height above the ground under the drone
    height_over_target_m: float  # height above the ground at the animal


def _limit_norm(x: float, y: float, limit: float) -> tuple[float, float]:
    m = math.hypot(x, y)
    if m <= limit or m == 0.0:
        return x, y
    k = limit / m
    return x * k, y * k


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class FollowController:
    def __init__(self, cfg: ControllerCfg, safety: SafetyCfg):
        self.cfg = cfg
        self.safety = safety
        self.i_n = 0.0  # integral term, already multiplied by Ki (m/s)
        self.i_e = 0.0
        self.prev_vn = 0.0
        self.prev_ve = 0.0
        self.prev_t: float | None = None
        self.standoff_bearing: float | None = None  # bearing animal -> drone, degrees

    def reset(self, vn_now: float = 0.0, ve_now: float = 0.0) -> None:
        """Called on every mode change. Starts the acceleration limit from the current velocity."""
        self.i_n = self.i_e = 0.0
        self.prev_vn, self.prev_ve = vn_now, ve_now
        self.prev_t = None

    def setpoint(self, animal_n: float, animal_e: float, drone_n: float, drone_e: float) -> tuple[float, float]:
        """Standoff point: along the drone's current bearing from the animal, never overhead."""
        dn, de = drone_n - animal_n, drone_e - animal_e
        if math.hypot(dn, de) >= self.cfg.bearing_hold_m or self.standoff_bearing is None:
            if math.hypot(dn, de) < 1e-6:
                # exactly overhead with no history: any bearing is as good as another
                self.standoff_bearing = 180.0
            else:
                self.standoff_bearing = bearing_deg(dn, de)
        b = math.radians(self.standoff_bearing)
        r = self.cfg.standoff_m
        return animal_n + r * math.cos(b), animal_e + r * math.sin(b)

    def _vertical(self, agl_m: float) -> float:
        """Zero inside the altitude band; otherwise steer back into it (NED: negative = up)."""
        if agl_m < self.safety.alt_min_m:
            return -min(self.cfg.vz_kp * (self.safety.alt_min_m - agl_m), self.cfg.vz_max_mps)
        if agl_m > self.safety.alt_max_m:
            return min(self.cfg.vz_kp * (agl_m - self.safety.alt_max_m), self.cfg.vz_max_mps)
        return 0.0

    def _yaw_rate(self, drone: DroneState, target: TargetEstimate) -> float:
        dn, de = target.n - drone.n, target.e - drone.e
        if math.hypot(dn, de) < self.cfg.bearing_hold_m:
            return 0.0
        err = wrap180(bearing_deg(dn, de) - drone.yaw_deg)
        lim = self.cfg.yaw_rate_max_dps
        return _clamp(self.cfg.yaw_kp * err, -lim, lim)

    def gimbal_pitch(self, drone: DroneState, target: TargetEstimate, box_offset_deg: float | None) -> float:
        """-atan(height / distance), plus a fraction of the box's angular offset below centre."""
        dist = math.hypot(target.n - drone.n, target.e - drone.e)
        pitch = -math.degrees(math.atan2(max(drone.height_over_target_m, 0.0), dist))
        if box_offset_deg is not None:
            pitch -= self.cfg.gimbal_box_gain * box_offset_deg
        return _clamp(pitch, self.cfg.gimbal_min_deg, self.cfg.gimbal_max_deg)

    def step(self, drone: DroneState, target: TargetEstimate, box_offset_deg: float | None = None) -> Command:
        c = self.cfg
        dt = 1.0 / c.rate_hz if self.prev_t is None else _clamp(drone.t - self.prev_t, 0.0, 2.0 / c.rate_hz)
        self.prev_t = drone.t

        sn, se = self.setpoint(target.n, target.e, drone.n, drone.e)
        en, ee = sn - drone.n, se - drone.e

        # candidate integral; kept only if the output is not saturated (anti-windup)
        i_n, i_e = _limit_norm(self.i_n + c.ki * en * dt, self.i_e + c.ki * ee * dt, c.integrator_limit_mps)
        vn = target.vn + c.kp * en + i_n
        ve = target.ve + c.kp * ee + i_e
        if math.hypot(vn, ve) <= c.v_max_mps:
            self.i_n, self.i_e = i_n, i_e
        vn, ve = _limit_norm(vn, ve, c.v_max_mps)

        # acceleration limit so the drone doesn't lurch
        dvn, dvel = _limit_norm(vn - self.prev_vn, ve - self.prev_ve, c.a_max_mps2 * dt)
        vn, ve = self.prev_vn + dvn, self.prev_ve + dvel
        self.prev_vn, self.prev_ve = vn, ve

        return Command(
            t=drone.t,
            vn=vn,
            ve=ve,
            vd=self._vertical(drone.agl_m),
            yaw_rate=self._yaw_rate(drone, target),
            gimbal_pitch=self.gimbal_pitch(drone, target, box_offset_deg),
        )

    def hover(self, t: float, gimbal_pitch: float) -> Command:
        """Lost: zero velocity, yaw held, gimbal held on the last bearing."""
        self.prev_vn = self.prev_ve = 0.0
        self.prev_t = t
        return Command(t, 0.0, 0.0, 0.0, 0.0, gimbal_pitch)
