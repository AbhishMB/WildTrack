import math
from pathlib import Path

import pytest

from wildtrack import config
from wildtrack.models import Attitude, Pose, Telemetry

CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture
def cfg():
    return config.load(CONFIG)


def telemetry(
    lat=12.2958,
    lon=76.6394,
    alt=60.0,
    yaw=0.0,
    gimbal_pitch=-55.0,
    gimbal_yaw=None,
    vel=(0.0, 0.0, 0.0),
    battery=80.0,
    sats=17,
    vs=False,
    home=(12.2958, 76.6394),
    seq=1,
) -> Telemetry:
    return Telemetry.from_msg(
        {
            "type": "telemetry",
            "seq": seq,
            "t_phone_ms": 0,
            "lat": lat,
            "lon": lon,
            "alt_rel_m": alt,
            "att": {"pitch": 0.0, "roll": 0.0, "yaw": yaw},
            "vel_ned": {"n": vel[0], "e": vel[1], "d": vel[2]},
            "gimbal": {"pitch": gimbal_pitch, "roll": 0.0, "yaw": yaw if gimbal_yaw is None else gimbal_yaw},
            "battery_pct": battery,
            "gps_sats": sats,
            "vs_enabled": vs,
            "flight_mode": "P",
            "home": None if home is None else {"lat": home[0], "lon": home[1]},
        }
    )


def pose(lat=12.2958, lon=76.6394, alt=60.0, pitch=-90.0, yaw=0.0, roll=0.0, t=0.0) -> Pose:
    return Pose(t, lat, lon, alt, Attitude(0.0, 0.0, yaw), Attitude(pitch, roll, yaw), (0.0, 0.0, 0.0))


def close(a, b, tol):
    return math.isclose(a, b, abs_tol=tol)
