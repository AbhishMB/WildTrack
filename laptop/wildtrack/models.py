"""Laptop data types (docs/SPEC.md, "Laptop types") and the telemetry wire message.

All times `t` are laptop monotonic seconds (time.monotonic()). Angles are degrees.
Local north/east coordinates (n, e) are metres from the home point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Attitude:
    pitch: float
    roll: float
    yaw: float


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float


@dataclass(frozen=True)
class Pose:
    """Aircraft and gimbal state at one instant."""

    t: float
    lat: float
    lon: float
    alt_rel: float  # metres above the takeoff point
    att: Attitude
    gimbal_att: Attitude  # north-east-down frame
    vel_ned: tuple[float, float, float]


@dataclass(frozen=True)
class Detection:
    frame_t: float  # estimated capture time: frame arrival - video lag
    track_id: int
    cls: str
    conf: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Fix:
    """One geolocated sighting with its 1-sigma horizontal error radius."""

    t: float
    lat: float
    lon: float
    n: float
    e: float
    sigma_m: float


@dataclass(frozen=True)
class TargetEstimate:
    t: float
    n: float
    e: float
    vn: float
    ve: float
    cov: np.ndarray = field(compare=False)  # 4x4 over [n, e, vn, ve]

    @property
    def sigma_m(self) -> float:
        """1-sigma horizontal position radius (sqrt of the mean positional variance)."""
        return math.sqrt(max(0.0, 0.5 * (self.cov[0, 0] + self.cov[1, 1])))

    @property
    def speed(self) -> float:
        return math.hypot(self.vn, self.ve)

    @property
    def heading_deg(self) -> float:
        return math.degrees(math.atan2(self.ve, self.vn)) % 360.0


@dataclass(frozen=True)
class Command:
    t: float
    vn: float  # ground-frame velocity, m/s
    ve: float
    vd: float  # positive = down
    yaw_rate: float  # deg/s, positive = clockwise seen from above
    gimbal_pitch: float  # deg, negative = down


def _num(v) -> float:
    """JSON null (value not yet known on the phone) becomes NaN."""
    return math.nan if v is None else float(v)


@dataclass(frozen=True)
class Telemetry:
    """The phone's 10 Hz telemetry message, parsed (docs/SPEC.md, "Wire format")."""

    seq: int
    t_phone_ms: int
    lat: float
    lon: float
    alt_rel_m: float
    att: Attitude
    vel_ned: tuple[float, float, float]
    gimbal: Attitude
    battery_pct: float
    gps_sats: int
    vs_enabled: bool
    flight_mode: str
    home: LatLon | None

    @staticmethod
    def from_msg(m: dict) -> Telemetry:
        if m.get("type") != "telemetry":
            raise ValueError("not a telemetry message")
        att, gim, vel = m["att"], m["gimbal"], m["vel_ned"]
        home = m.get("home")
        home_ll = None
        if home and home.get("lat") is not None and home.get("lon") is not None:
            home_ll = LatLon(float(home["lat"]), float(home["lon"]))
        return Telemetry(
            seq=int(m["seq"]),
            t_phone_ms=int(m["t_phone_ms"]),
            lat=_num(m["lat"]),
            lon=_num(m["lon"]),
            alt_rel_m=_num(m["alt_rel_m"]),
            att=Attitude(_num(att["pitch"]), _num(att["roll"]), _num(att["yaw"])),
            vel_ned=(_num(vel["n"]), _num(vel["e"]), _num(vel["d"])),
            gimbal=Attitude(_num(gim["pitch"]), _num(gim["roll"]), _num(gim["yaw"])),
            battery_pct=_num(m["battery_pct"]),
            gps_sats=int(m["gps_sats"] or 0),
            vs_enabled=bool(m["vs_enabled"]),
            flight_mode=str(m["flight_mode"]),
            home=home_ll,
        )

    @property
    def position_valid(self) -> bool:
        return all(math.isfinite(x) for x in (self.lat, self.lon, self.alt_rel_m)) and not (
            self.lat == 0.0 and self.lon == 0.0
        )

    def pose(self, t: float) -> Pose:
        return Pose(t, self.lat, self.lon, self.alt_rel_m, self.att, self.gimbal, self.vel_ned)
