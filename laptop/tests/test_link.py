import json
import math

import pytest

from wildtrack.link import PoseRing
from wildtrack.models import Telemetry

from .conftest import telemetry

SPEC_EXAMPLE = """{"type": "telemetry", "seq": 1042, "t_phone_ms": 1730000000123,
 "lat": 12.2958, "lon": 76.6394, "alt_rel_m": 61.2,
 "att": {"pitch": -2.1, "roll": 0.8, "yaw": 134.5},
 "vel_ned": {"n": 1.2, "e": -0.4, "d": 0.0},
 "gimbal": {"pitch": -58.0, "roll": 0.0, "yaw": 131.0},
 "battery_pct": 72, "gps_sats": 17,
 "vs_enabled": true, "flight_mode": "P",
 "home": {"lat": 12.2901, "lon": 76.6402}}"""


def test_parses_the_spec_message():
    t = Telemetry.from_msg(json.loads(SPEC_EXAMPLE))
    assert t.seq == 1042 and t.alt_rel_m == 61.2 and t.gimbal.pitch == -58.0
    assert t.vel_ned == (1.2, -0.4, 0.0) and t.vs_enabled and t.home.lat == 12.2901
    assert t.position_valid


def test_nulls_before_gps_fix():
    m = json.loads(SPEC_EXAMPLE)
    m["lat"] = m["lon"] = None
    m["home"] = None
    t = Telemetry.from_msg(m)
    assert math.isnan(t.lat) and t.home is None and not t.position_valid


def test_rejects_other_messages():
    with pytest.raises(ValueError):
        Telemetry.from_msg({"type": "event"})


def test_ring_interpolates():
    r = PoseRing(5.0)
    r.push(0.0, telemetry(alt=50.0, gimbal_pitch=-40.0, yaw=10.0))
    r.push(0.1, telemetry(alt=60.0, gimbal_pitch=-60.0, yaw=30.0))
    p = r.at(0.05, 0.2)
    assert p.alt_rel == pytest.approx(55.0)
    assert p.gimbal_att.pitch == pytest.approx(-50.0)
    assert p.gimbal_att.yaw == pytest.approx(20.0)


def test_ring_interpolates_yaw_across_180():
    r = PoseRing(5.0)
    r.push(0.0, telemetry(yaw=170.0))
    r.push(0.1, telemetry(yaw=-170.0))
    assert abs(r.at(0.05, 0.2).att.yaw) == pytest.approx(180.0)


def test_ring_limits():
    r = PoseRing(1.0)
    for k in range(30):
        r.push(k * 0.1, telemetry(alt=float(k)))
    assert len(r) <= 11  # one second kept
    assert r.at(10.0, 0.2) is None  # far in the future
    assert r.at(2.95, 0.2).alt_rel == pytest.approx(29.0)  # just past the newest: hold it
    assert r.at(0.0, 0.2) is None  # dropped from the buffer
