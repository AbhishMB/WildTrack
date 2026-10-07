"""Safety behaviour of the Autopilot core, driven directly with telemetry and detections."""

import pytest

from wildtrack.core import Autopilot
from wildtrack.fsm import Mode
from wildtrack.models import Detection

from .conftest import telemetry

W, H = 1280, 720
# drone 40 m south-west-ish of an elephant seen in the middle of the frame at -56 deg
BOX = (620.0, 330.0, 660.0, 390.0)


class Rig:
    def __init__(self, cfg, **tel):
        self.ap = Autopilot(cfg, None)
        self.t = 0.0
        self.tel = dict(gimbal_pitch=-56.0, **tel)
        self.sent: list[dict] = []

    def step(self, dets=None, n=1, **tel_changes):
        self.tel.update(tel_changes)
        for _ in range(n):
            self.t += 0.1
            self.ap.on_telemetry(telemetry(**self.tel), self.t)
            if dets is not None:
                frame_t = self.t - 0.3
                self.ap.on_detections([Detection(frame_t, *d) for d in dets], W, H, self.t)
            self.sent += self.ap.tick(self.t)

    def kinds(self):
        return [m["type"] for m in self.sent]


ELEPHANT = (1, "elephant", 0.9, BOX)
PERSON = (2, "person", 0.95, (100.0, 300.0, 120.0, 360.0))


def locked_rig(cfg, **tel):
    r = Rig(cfg, **tel)
    r.step([ELEPHANT], n=6)
    assert r.ap.operator("lock", r.t, 1)["ok"]
    r.step([ELEPHANT], n=3)
    return r


def test_lock_engage_and_fly(cfg):
    r = locked_rig(cfg)
    assert r.ap.fsm.mode == Mode.LOCKED and r.ap.kf.initialized
    reply = r.ap.operator("engage", r.t)
    assert reply["ok"], reply
    r.step([ELEPHANT], n=5, vs=True)
    assert r.ap.fsm.mode == Mode.AUTO
    assert r.kinds()[0] == "vs_enable"
    assert r.kinds().count("vel") == 5 and "gimbal" in r.kinds()


def test_person_is_refused_even_when_clicked(cfg):
    r = Rig(cfg)
    r.step([ELEPHANT, PERSON], n=6)
    reply = r.ap.operator("lock", r.t, 2)
    assert not reply["ok"] and r.ap.fsm.mode == Mode.MANUAL


def test_release_always_sends_vs_disable(cfg):
    r = Rig(cfg)
    r.step(n=1)
    r.ap.operator("release", r.t)
    assert {"type": "vs_disable"} in r.ap.outbox


def test_engage_refused_with_few_satellites(cfg):
    r = locked_rig(cfg, sats=8)
    reply = r.ap.operator("engage", r.t)
    assert not reply["ok"] and "GPS" in reply["message"]


def test_dji_pause_drops_to_manual_and_stops_commands(cfg):
    r = locked_rig(cfg)
    r.ap.operator("engage", r.t)
    r.step([ELEPHANT], n=3, vs=True)
    r.ap.on_phone_event("vs_exit", "RC_PAUSE_STOP", r.t)
    assert r.ap.fsm.mode == Mode.MANUAL
    assert any("Pause" in a.text for a in r.ap.alerts)
    r.sent.clear()
    r.step([ELEPHANT], n=5, vs=False)
    assert "vel" not in r.kinds() and "vs_enable" not in r.kinds()  # never re-enabled


@pytest.mark.parametrize(
    "change,word",
    [({"battery": 30.0}, "battery"), ({"lat": 12.3058}, "home")],  # 0.01 deg north ~ 1.1 km
)
def test_hand_back_conditions(cfg, change, word):
    r = locked_rig(cfg)
    r.ap.operator("engage", r.t)
    r.step([ELEPHANT], n=3, vs=True)
    r.sent.clear()
    r.step([ELEPHANT], n=1, **change)
    assert r.ap.fsm.mode == Mode.MANUAL
    assert "vs_disable" in r.kinds()
    assert any(word in a.text for a in r.ap.alerts)


def test_link_loss_hands_back(cfg):
    r = locked_rig(cfg)
    r.ap.operator("engage", r.t)
    r.step([ELEPHANT], n=3, vs=True)
    r.t += 1.0  # no telemetry for a second
    r.ap.tick(r.t)
    assert r.ap.fsm.mode == Mode.MANUAL


def test_telemetry_showing_vs_off_after_grace_hands_back(cfg):
    r = locked_rig(cfg)
    r.ap.operator("engage", r.t)
    r.step([ELEPHANT], n=5, vs=False)  # inside the grace period: still AUTO
    assert r.ap.fsm.mode == Mode.AUTO
    r.step([ELEPHANT], n=20, vs=False)
    assert r.ap.fsm.mode == Mode.MANUAL


def test_lost_then_hand_back(cfg):
    r = locked_rig(cfg)
    r.ap.operator("engage", r.t)
    r.step([ELEPHANT], n=3, vs=True)
    r.step([], n=25, vs=True)  # 2.5 s without the animal
    assert r.ap.fsm.mode == Mode.LOST
    assert r.sent[-1]["type"] == "vel" and r.sent[-1]["vn"] == 0 and r.sent[-1]["ve"] == 0
    r.step([], n=80, vs=True)  # 10 s since the last fix
    assert r.ap.fsm.mode == Mode.MANUAL and "vs_disable" in r.kinds()


def test_locked_track_lost_returns_to_manual(cfg):
    r = locked_rig(cfg)
    r.step([], n=int(cfg.lock.track_lost_s * 10) + 2)
    assert r.ap.fsm.mode == Mode.MANUAL


def test_publishes_position_with_error_radius(cfg):
    r = locked_rig(cfg)
    a = r.ap.animal(r.t)
    assert a["cls"] == "elephant" and a["sigma_m"] > cfg.geo.sigma_gps_m and a["locked"]
    assert a["age_s"] < 0.5 and len(a["trail"]) >= 3
