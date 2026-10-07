import dataclasses
import math

import numpy as np
import pytest

from wildtrack import geo
from wildtrack.models import Attitude, LatLon

from .conftest import close, pose

W, H = 1280, 720
HOME = LatLon(12.2958, 76.6394)


def locate(cfg, p, u=W / 2, v=H / 2, terrain=None, z_home=0.0):
    return geo.geolocate(p, u, v, W, H, cfg.camera.hfov_deg, HOME, z_home, terrain, cfg.geo)


def test_nadir_centre_pixel_is_below_the_drone(cfg):
    f = locate(cfg, pose(alt=60.0, pitch=-90.0))
    assert close(f.n, 0.0, 1e-6) and close(f.e, 0.0, 1e-6)
    # straight down: only GPS and A * sigma_alpha remain
    expect = math.hypot(cfg.geo.sigma_gps_m, 60.0 * math.radians(cfg.geo.sigma_att_deg))
    assert close(f.sigma_m, expect, 1e-6)


@pytest.mark.parametrize("yaw,dn,de", [(0, 60, 0), (90, 0, 60), (180, -60, 0), (-90, 0, -60)])
def test_45_degrees_lands_one_height_away_along_yaw(cfg, yaw, dn, de):
    f = locate(cfg, pose(alt=60.0, pitch=-45.0, yaw=yaw))
    assert close(f.n, dn, 1e-6) and close(f.e, de, 1e-6)


def test_pixel_to_ground_and_back(cfg):
    """geolocate is the exact inverse of projecting a ground point into the image."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        att = Attitude(float(rng.uniform(-85, -35)), float(rng.uniform(-3, 3)), float(rng.uniform(-180, 180)))
        p = dataclasses.replace(pose(alt=float(rng.uniform(40, 80))), gimbal_att=att)
        u, v = float(rng.uniform(0, W)), float(rng.uniform(H * 0.3, H))
        f = locate(cfg, p, u, v)
        if f is None:
            continue
        uv = geo.project(np.array([f.n, f.e, p.alt_rel]), W, H, cfg.camera.hfov_deg, att)
        assert close(uv[0], u, 1e-6) and close(uv[1], v, 1e-6)


def test_ray_at_or_above_horizon_gives_no_fix(cfg):
    assert locate(cfg, pose(pitch=0.0)) is None
    assert locate(cfg, pose(pitch=10.0)) is None
    assert locate(cfg, pose(pitch=-20.0), v=0.0) is None  # top row is above the horizon


def test_too_far_or_too_flat_gives_no_fix(cfg):
    assert locate(cfg, pose(alt=60.0, pitch=-4.0)) is None  # flatter than min_depression_deg
    assert locate(cfg, pose(alt=60.0, pitch=-5.5)) is None  # 624 m > max_range_m
    f = locate(cfg, pose(alt=60.0, pitch=-7.0))  # 489 m
    assert f is not None and close(math.hypot(f.n, f.e), 60.0 / math.tan(math.radians(7.0)), 1e-6)


def test_error_radius_matches_spec_example(cfg):
    """60 m up, 40 m out: 1 deg attitude error adds ~1.5 m, 5 m height error ~3.4 m."""
    alpha = math.atan2(60.0, 40.0)
    only_att = dataclasses.replace(cfg.geo, sigma_gps_m=0.0, sigma_att_deg=1.0, sigma_alt_m=0.0)
    only_hgt = dataclasses.replace(cfg.geo, sigma_gps_m=0.0, sigma_att_deg=0.0, sigma_alt_m=5.0)
    assert close(geo.sigma_position(60.0, alpha, only_att), 1.5, 0.05)
    assert close(geo.sigma_position(60.0, alpha, only_hgt), 3.33, 0.1)


def test_error_shrinks_as_camera_looks_down(cfg):
    s = [geo.sigma_position(60.0, math.radians(a), cfg.geo) for a in (20, 40, 60, 80)]
    assert s == sorted(s, reverse=True)


def test_terrain_iteration_on_a_slope(cfg):
    """Ground rises 0.1 m per metre north. Looking north at 45 deg from 60 m: hit at 60 / 1.1 m."""
    k = 0.1

    def terrain(lat, lon):
        n, _ = geo.latlon_to_ned(LatLon(lat, lon), HOME, cfg.geo.earth_radius_m)
        return k * n

    f = locate(cfg, pose(alt=60.0, pitch=-45.0), terrain=terrain, z_home=0.0)
    assert close(f.n, 60.0 / (1 + k), 0.05)


def test_flat_dem_equals_no_dem(cfg):
    a = locate(cfg, pose(pitch=-50.0, yaw=30.0), u=900, v=600)
    b = locate(cfg, pose(pitch=-50.0, yaw=30.0), u=900, v=600, terrain=lambda lat, lon: 812.0, z_home=812.0)
    assert close(a.n, b.n, 1e-9) and close(a.e, b.e, 1e-9)


def test_latlon_round_trip(cfg):
    R = cfg.geo.earth_radius_m
    for n, e in [(0, 0), (500, -300), (-1000, 1000)]:
        p = geo.ned_to_latlon(n, e, HOME, R)
        n2, e2 = geo.latlon_to_ned(p, HOME, R)
        assert close(n, n2, 1e-6) and close(e, e2, 1e-6)
    # 1 m north is about 9e-6 degrees of latitude
    assert close(geo.ned_to_latlon(1.0, 0.0, HOME, R).lat - HOME.lat, 8.993e-6, 1e-8)


def test_box_point(cfg):
    box = (100, 200, 140, 260)
    assert geo.box_point(box, -50.0, cfg.geo.nadir_pitch_deg) == (120, 260)  # oblique: bottom-centre
    assert geo.box_point(box, -88.0, cfg.geo.nadir_pitch_deg) == (120, 230)  # straight down: centre


def test_dms():
    assert geo.to_dms(12.2958, "N", "S") == "12°17'44.88\"N"
    assert geo.to_dms(-0.5, "N", "S") == "0°30'00.00\"S"


def test_bearing_and_wrap():
    assert geo.bearing_deg(1, 0) == 0
    assert geo.bearing_deg(0, 1) == 90
    assert geo.bearing_deg(-1, 0) == 180
    assert geo.wrap180(190) == -170
    assert geo.wrap180(-190) == 170
