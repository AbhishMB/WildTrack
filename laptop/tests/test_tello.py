import asyncio
import dataclasses
import math
import socket

import pytest

from wildtrack.config import override
from wildtrack.geo import latlon_to_ned
from wildtrack.models import LatLon, Telemetry
from wildtrack.sim.fake_tello import FakeTello
from wildtrack.tello import DeadReckoner, TelloLink, ground_to_rc, parse_state, telemetry_message


def free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def tcfg(cfg):
    c = override(cfg, "link.backend", "tello")
    c = override(c, "tello.host", "127.0.0.1")
    c = override(c, "tello.cmd_port", free_udp_port())
    c = override(c, "tello.state_port", free_udp_port())
    c = override(c, "tello.connect_retry_s", 0.1)
    c = override(c, "tello.state_timeout_s", 1.0)
    return c


# ------------------------------------------------------------------ pure parts


def test_parse_state():
    st = parse_state("pitch:1;roll:-2;yaw:-30;vgx:5;vgy:0;vgz:-1;templ:60;temph:62;tof:100;h:90;bat:87;baro:12.5;time:3;agx:0.0;\r\n")
    assert st["yaw"] == -30 and st["h"] == 90 and st["bat"] == 87 and st["baro"] == 12.5
    assert parse_state("garbage;;x:y;h:5") == {"h": 5.0}


@pytest.mark.parametrize(
    "vn,ve,heading,expect",  # expect (right, forward) in m/s
    [
        (1.0, 0.0, 0.0, (0.0, 1.0)),     # facing north, go north: forward
        (0.0, 1.0, 0.0, (1.0, 0.0)),     # facing north, go east: right
        (0.0, 1.0, 90.0, (0.0, 1.0)),    # facing east, go east: forward
        (1.0, 0.0, 90.0, (-1.0, 0.0)),   # facing east, go north: left
        (-1.0, 0.0, 0.0, (0.0, -1.0)),   # facing north, go south: backward
        (1.0, 1.0, 45.0, (0.0, math.sqrt(2))),
    ],
)
def test_ground_to_rc_axis_mapping(cfg, vn, ve, heading, expect):
    t = dataclasses.replace(cfg.tello, rc_per_mps=10.0)
    a, b, c, d = ground_to_rc(vn, ve, 0.0, 0.0, heading, t)
    assert (a, b) == (round(expect[0] * 10), round(expect[1] * 10))
    assert c == 0 and d == 0


def test_ground_to_rc_vertical_yaw_and_clipping(cfg):
    t = dataclasses.replace(cfg.tello, rc_per_mps=30.0, yaw_dps_per_unit=1.0)
    assert ground_to_rc(0, 0, -0.5, 0, 0, t)[2] == 15       # vd negative = up = positive rc
    assert ground_to_rc(0, 0, 0.5, 0, 0, t)[2] == -15
    assert ground_to_rc(0, 0, 0, 20.0, 0, t)[3] == 20       # clockwise
    assert ground_to_rc(50.0, 0, 0, 500.0, 0, t) == (0, 100, 0, 100)


def test_dead_reckoning_in_takeoff_frame(cfg):
    t = dataclasses.replace(cfg.tello, takeoff_heading_deg=90.0, state_speed_mps=0.1)
    dr = DeadReckoner(t)
    dr.update({"vgx": 10, "vgy": 0}, 0.0)  # 1 m/s forward; the nose pointed east at takeoff
    for k in range(1, 11):
        dr.update({"vgx": 10, "vgy": 0}, k * 0.1)
    assert (dr.n, dr.e) == pytest.approx((0.0, 1.0), abs=1e-9)
    dr.reset()
    assert (dr.n, dr.e) == (0.0, 0.0)


def test_dead_reckoning_ignores_gaps(cfg):
    dr = DeadReckoner(cfg.tello)
    dr.update({"vgx": 10}, 0.0)
    dr.update({"vgx": 10}, 30.0)  # 30 s dropout: capped at 0.5 s of travel
    assert dr.n == pytest.approx(0.5)


def test_telemetry_message_matches_the_wire_format(cfg):
    dr = DeadReckoner(cfg.tello)
    dr.n, dr.e = 3.0, -4.0
    msg = telemetry_message({"pitch": 4.0, "roll": 1.0, "yaw": -20.0, "h": 150.0, "bat": 77.0}, dr, cfg, True, 7, 123)
    tel = Telemetry.from_msg(msg)  # the same parser the DJI path uses
    assert tel.alt_rel_m == 1.5 and tel.battery_pct == 77 and tel.vs_enabled and tel.gps_sats == 0
    assert tel.gimbal.pitch == pytest.approx(cfg.tello.camera_pitch_deg + 4.0)  # camera follows the airframe
    assert tel.att.yaw == pytest.approx(-20.0 + cfg.tello.takeoff_heading_deg)
    n, e = latlon_to_ned(LatLon(tel.lat, tel.lon), LatLon(*cfg.tello.home), cfg.geo.earth_radius_m)
    assert (n, e) == pytest.approx((3.0, -4.0), abs=1e-6)
    assert math.isnan(tel.vel_ned[2]) and tel.home.lat == cfg.tello.home[0]


# ------------------------------------------------------------------ link against the fake drone


class Rig:
    def __init__(self, tcfg):
        self.cfg = tcfg
        self.tel: list[Telemetry] = []
        self.events: list[tuple[str, str]] = []
        self.states: list[bool] = []
        self.fake = FakeTello(tcfg.tello)
        self.link = TelloLink(tcfg, lambda t, m, now: self.tel.append(t), lambda n, r, now: self.events.append((n, r)),
                              lambda up, now: self.states.append(up))
        self.task: asyncio.Task | None = None

    async def __aenter__(self):
        await self.fake.start()
        self.task = asyncio.create_task(self.link.run())
        for _ in range(100):
            if self.link.connected and self.tel:
                break
            await asyncio.sleep(0.05)
        assert self.link.connected, "handshake failed"
        return self

    async def __aexit__(self, *a):
        self.task.cancel()
        self.fake.stop()


def run(coro):
    return asyncio.run(coro)


def test_handshake_telemetry_and_takeoff(tcfg):
    async def go():
        async with Rig(tcfg) as r:
            assert r.fake.log[:2] == ["command", "streamon"] and r.fake.streaming
            assert r.states == [True]
            assert await r.link.send({"type": "takeoff"})
            await asyncio.sleep(0.4)
            assert r.fake.flying and r.tel[-1].alt_rel_m > 0.5
            assert r.tel[-1].battery_pct == 90 and r.tel[-1].position_valid
            assert not await r.link.send({"type": "bogus"})

    run(go())


def test_velocity_command_becomes_a_forward_stick_and_moves_the_drone(tcfg):
    async def go():
        async with Rig(tcfg) as r:
            await r.link.send({"type": "takeoff"})
            await r.link.send({"type": "vs_enable"})
            for _ in range(15):  # 1.5 s of "go north at 1 m/s", the nose points north
                await r.link.send({"type": "vel", "vn": 1.0, "ve": 0.0, "vd": 0.0, "yaw_rate_dps": 0.0})
                await asyncio.sleep(0.1)
            assert r.fake.rc[1] == round(1.0 * tcfg.tello.rc_per_mps) and r.fake.rc[0] == 0
            n, e = latlon_to_ned(LatLon(r.tel[-1].lat, r.tel[-1].lon), LatLon(*tcfg.tello.home), tcfg.geo.earth_radius_m)
            assert n == pytest.approx(r.fake.x, abs=0.5) and n > 0.8     # dead reckoning follows the real motion
            assert abs(e) < 0.2
            assert r.tel[-1].vs_enabled and ("vs_enabled", "") in r.events

    run(go())


def test_speed_is_clamped_by_the_link(tcfg):
    async def go():
        async with Rig(tcfg) as r:
            await r.link.send({"type": "takeoff"})
            await r.link.send({"type": "vs_enable"})
            await r.link.send({"type": "vel", "vn": 50.0, "ve": 0.0, "vd": 0.0, "yaw_rate_dps": 0.0})
            await asyncio.sleep(0.2)
            assert r.fake.rc[1] <= round(tcfg.tello.max_speed_mps * tcfg.tello.rc_per_mps)

    run(go())


def test_laptop_going_quiet_hovers_then_hands_back(tcfg):
    c = override(override(tcfg, "tello.watchdog_hover_s", 0.2), "tello.watchdog_handback_s", 0.8)

    async def go():
        async with Rig(c) as r:
            await r.link.send({"type": "takeoff"})
            await r.link.send({"type": "vs_enable"})
            await r.link.send({"type": "vel", "vn": 1.0, "ve": 0.0, "vd": 0.0, "yaw_rate_dps": 0.0})
            await asyncio.sleep(0.2)
            assert r.fake.rc[1] > 0
            await asyncio.sleep(0.6)  # no more commands
            assert r.fake.rc == (0, 0, 0, 0)  # hovering
            assert ("watchdog_hover", "") in r.events
            await asyncio.sleep(0.8)
            assert ("watchdog_handback", "") in r.events and ("vs_exit", "WATCHDOG") in r.events
            assert not r.link.vs_enabled and r.tel[-1].vs_enabled is False

    run(go())


def test_vs_disable_is_reported_as_vs_exit_and_hovers(tcfg):
    async def go():
        async with Rig(tcfg) as r:
            await r.link.send({"type": "takeoff"})
            await r.link.send({"type": "vs_enable"})
            await r.link.send({"type": "vel", "vn": 1.0, "ve": 0.0, "vd": 0.0, "yaw_rate_dps": 0.0})
            await asyncio.sleep(0.15)
            await r.link.send({"type": "vs_disable"})
            await asyncio.sleep(0.15)
            assert r.fake.rc == (0, 0, 0, 0) and ("vs_exit", "LAPTOP_REQUEST") in r.events

    run(go())


def test_manual_sticks_work_only_outside_auto_and_go_stale(tcfg):
    async def go():
        async with Rig(tcfg) as r:
            await r.link.send({"type": "takeoff"})
            await r.link.send({"type": "rc", "fwd": 1.0, "right": -1.0, "up": 0.5, "yaw": 0.0})
            await asyncio.sleep(0.15)
            k = round(100 * tcfg.tello.manual_speed_frac)
            assert r.fake.rc == (-k, k, round(k * 0.5), 0)
            await asyncio.sleep(tcfg.tello.rc_stale_s + 0.2)  # key released / dashboard closed: hover
            assert r.fake.rc == (0, 0, 0, 0)
            await r.link.send({"type": "vs_enable"})
            assert not await r.link.send({"type": "rc", "fwd": 1.0})  # AUTO owns the sticks

    run(go())


def test_land_and_emergency(tcfg):
    async def go():
        async with Rig(tcfg) as r:
            await r.link.send({"type": "takeoff"})
            await r.link.send({"type": "vs_enable"})
            assert await r.link.send({"type": "land"})
            assert not r.fake.flying and not r.link.vs_enabled
            await r.link.send({"type": "takeoff"})
            await r.link.send({"type": "emergency"})
            await asyncio.sleep(0.1)
            assert r.fake.emergency and not r.fake.flying

    run(go())


def test_reconnects_after_the_drone_goes_silent(tcfg):
    async def go():
        async with Rig(tcfg) as r:
            r.fake.stop()
            await asyncio.sleep(1.6)
            assert r.states[:2] == [True, False] and not r.link.connected
            r.fake = FakeTello(tcfg.tello)
            await r.fake.start()
            for _ in range(60):
                if r.link.connected:
                    break
                await asyncio.sleep(0.1)
            assert r.link.connected and r.states[-1] is True
            r.fake.stop()

    run(go())
