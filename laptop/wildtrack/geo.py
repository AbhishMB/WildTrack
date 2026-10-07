"""Pixel -> camera ray -> ground point -> latitude/longitude, with a 1-sigma error radius.

Implements docs/SPEC.md, "Animal geolocation". Pure: no I/O. Terrain comes in as a
function (dem.py loads the tiles), so tests can pass any surface they like.

Frames: camera axes are (forward, right, down); world axes are north-east-down (NED).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .config import GeoCfg
from .models import Attitude, Fix, LatLon, Pose

# (lat, lon) -> terrain height in metres, or None where unknown. Any vertical datum
# works, because only differences against the home point's height are used.
TerrainFn = Callable[[float, float], Optional[float]]


def focal_px(width: float, hfov_deg: float) -> float:
    """f = (W/2) / tan(HFOV/2), in pixels."""
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def ray_cam(u: float, v: float, width: float, height: float, f: float) -> np.ndarray:
    """Step 1: ray through pixel (u, v) in camera axes (forward, right, down)."""
    return np.array([1.0, (u - width / 2.0) / f, (v - height / 2.0) / f])


def rot_zyx(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll): camera axes -> NED."""
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def ray_ned(u: float, v: float, width: float, height: float, hfov_deg: float, gimbal: Attitude) -> np.ndarray:
    """Step 2: the pixel's ray in north-east-down."""
    d = ray_cam(u, v, width, height, focal_px(width, hfov_deg))
    return rot_zyx(gimbal.yaw, gimbal.pitch, gimbal.roll) @ d


def project(point_ned: np.ndarray, width: float, height: float, hfov_deg: float, gimbal: Attitude):
    """Inverse of ray_ned: NED offset from the camera -> pixel (u, v), or None if behind the camera."""
    d = rot_zyx(gimbal.yaw, gimbal.pitch, gimbal.roll).T @ np.asarray(point_ned, dtype=float)
    if d[0] <= 1e-6:
        return None
    f = focal_px(width, hfov_deg)
    return width / 2.0 + f * d[1] / d[0], height / 2.0 + f * d[2] / d[0]


def box_point(box_xyxy, gimbal_pitch_deg: float, nadir_pitch_deg: float) -> tuple[float, float]:
    """Where the animal meets the ground: bottom-centre when oblique, centre when looking down."""
    x1, y1, x2, y2 = box_xyxy
    u = 0.5 * (x1 + x2)
    if gimbal_pitch_deg <= nadir_pitch_deg:
        return u, 0.5 * (y1 + y2)
    return u, y2


def ned_to_latlon(n: float, e: float, origin: LatLon, earth_radius_m: float) -> LatLon:
    """Step 5, flat-earth: accurate to a few cm over the few-km operating radius."""
    lat = origin.lat + math.degrees(n / earth_radius_m)
    lon = origin.lon + math.degrees(e / (earth_radius_m * math.cos(math.radians(origin.lat))))
    return LatLon(lat, lon)


def latlon_to_ned(p: LatLon, origin: LatLon, earth_radius_m: float) -> tuple[float, float]:
    n = math.radians(p.lat - origin.lat) * earth_radius_m
    e = math.radians(p.lon - origin.lon) * earth_radius_m * math.cos(math.radians(origin.lat))
    return n, e


def sigma_position(height_m: float, depression_rad: float, cfg: GeoCfg) -> float:
    """1-sigma horizontal error: GPS + attitude error + height error (SPEC, "Error radius")."""
    s2 = math.sin(depression_rad) ** 2
    att = height_m / s2 * math.radians(cfg.sigma_att_deg)
    hgt = cfg.sigma_alt_m / math.tan(depression_rad)
    return math.sqrt(cfg.sigma_gps_m**2 + att**2 + hgt**2)


@dataclass(frozen=True)
class GroundHit:
    dn: float  # horizontal offset from the drone, metres
    de: float
    height_m: float  # drone height above the ground at the hit point
    depression_rad: float  # ray angle below horizontal


def intersect_ground(
    d_ned: np.ndarray,
    drone: LatLon,
    alt_rel_m: float,
    z_home: float,
    terrain: TerrainFn | None,
    cfg: GeoCfg,
) -> GroundHit | None:
    """Steps 3-4: intersect the ray with the terrain, iterating on the DEM height at the hit point."""
    dn_, de_, dd_ = (float(x) for x in d_ned)
    if dd_ <= 0.0:
        return None  # at or above the horizon
    horiz = math.hypot(dn_, de_)
    depression = math.atan2(dd_, horiz)
    if depression < math.radians(cfg.min_depression_deg):
        return None
    z_terrain = z_home
    for i in range(cfg.terrain_iterations + 1):
        height = z_home + alt_rel_m - z_terrain
        if height <= 0.0:
            return None
        t = height / dd_
        dn, de = t * dn_, t * de_
        if i == cfg.terrain_iterations or terrain is None:
            break
        hit = ned_to_latlon(dn, de, drone, cfg.earth_radius_m)
        z = terrain(hit.lat, hit.lon)
        if z is None:
            break
        z_terrain = z
    if math.hypot(dn, de) > cfg.max_range_m:
        return None
    return GroundHit(dn, de, height, depression)


def geolocate(
    pose: Pose,
    u: float,
    v: float,
    width: int,
    height: int,
    hfov_deg: float,
    home: LatLon,
    z_home: float,
    terrain: TerrainFn | None,
    cfg: GeoCfg,
) -> Fix | None:
    """Pixel (u, v) of a frame captured at pose.t -> ground Fix, or None (no fix)."""
    d = ray_ned(u, v, width, height, hfov_deg, pose.gimbal_att)
    drone = LatLon(pose.lat, pose.lon)
    hit = intersect_ground(d, drone, pose.alt_rel, z_home, terrain, cfg)
    if hit is None:
        return None
    p = ned_to_latlon(hit.dn, hit.de, drone, cfg.earth_radius_m)
    n, e = latlon_to_ned(p, home, cfg.earth_radius_m)
    return Fix(pose.t, p.lat, p.lon, n, e, sigma_position(hit.height_m, hit.depression_rad, cfg))


def ground_height(lat: float, lon: float, z_home: float, terrain: TerrainFn | None) -> float:
    """Terrain height at a point, falling back to the home height (flat ground)."""
    if terrain is None:
        return z_home
    z = terrain(lat, lon)
    return z_home if z is None else z


def bearing_deg(dn: float, de: float) -> float:
    """Compass bearing of the vector (dn, de), 0 = north, 90 = east."""
    return math.degrees(math.atan2(de, dn)) % 360.0


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def to_dms(deg: float, pos: str, neg: str) -> str:
    h = pos if deg >= 0 else neg
    a = abs(deg)
    d = int(a)
    m_full = (a - d) * 60.0
    m = int(m_full)
    s = (m_full - m) * 60.0
    if s >= 59.995:
        s, m = 0.0, m + 1
    if m == 60:
        m, d = 0, d + 1
    return f"{d}°{m:02d}'{s:05.2f}\"{h}"
