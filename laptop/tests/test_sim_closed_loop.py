"""Build stage 3 acceptance, in the laptop-only simulator:

  follows an animal walking at 1.5 m/s with mean position error under 5 m;
  stays stable with tau up to 0.8 s;
  Lost -> hand-back works.
Plus the watchdog and Pause behaviour of the phone stand-in.
"""

import math

import pytest

from wildtrack.config import override
from wildtrack.sim.closed_loop import Scenario, run
from wildtrack.sim.drone import Watchdog


def modes(res):
    return [(src, dst, ev) for _, src, dst, ev in res.transitions]


def test_follows_walking_animal(cfg):
    res = run(cfg, Scenario(duration_s=120.0, animal_speed_mps=1.5))
    assert res.final_mode == "AUTO"
    assert res.standoff_err_mean < 5.0, res.summary()
    assert res.estimate_err_mean < 5.0, res.summary()


@pytest.mark.parametrize("tau", [0.4, 0.8])
def test_stable_with_delay(cfg, tau):
    res = run(cfg, Scenario(duration_s=120.0, animal_speed_mps=1.5, tau_s=tau))
    assert res.final_mode == "AUTO"
    assert res.standoff_err_max < 10.0, res.summary()


def test_step_response(cfg):
    """Animal standing still, drone starts 20 m outside the standoff: settles without big overshoot."""
    c = override(cfg, "sim.start_standoff_m", 60.0)
    res = run(c, Scenario(duration_s=60.0, animal_speed_mps=0.0), settle_s=30.0)
    assert res.standoff_err_mean < 3.0, res.summary()
    dists = [math.hypot(d[0] - a[0], d[1] - a[1]) for d, a in zip(res.trace.drone, res.trace.animal)]
    assert min(dists) > cfg.controller.standoff_m - 6.0  # never dives far inside the standoff


def test_short_occlusion_is_ridden_out(cfg):
    res = run(cfg, Scenario(duration_s=60.0, hide=(20.0, 24.0)))
    assert ("AUTO", "LOST", "no_fix") in modes(res)
    assert ("LOST", "AUTO", "reacquired") in modes(res)
    assert res.final_mode == "AUTO"


def test_lost_hands_back_after_ten_seconds(cfg):
    res = run(cfg, Scenario(duration_s=50.0, hide=(20.0, 45.0)))
    trans = {(src, dst): t for t, src, dst, _ in res.transitions}
    assert res.final_mode == "MANUAL"
    lost_to_manual = trans[("LOST", "MANUAL")]
    assert 29.0 < lost_to_manual < 31.5  # ~10 s after the last fix (hidden at 20 s, plus video lag)
    assert ("vs_exit", "MSDK_REQUEST") in [(n, r) for _, n, r in res.phone_events]
    assert not res.trace.vs_enabled[-1]


def test_pause_button(cfg):
    res = run(cfg, Scenario(duration_s=30.0, pause_at_s=15.0))
    assert ("AUTO", "MANUAL", "vs_exit") in modes(res)
    assert not any(res.trace.vs_enabled[int(16.0 / 0.01):])


def test_laptop_dies(cfg):
    res = run(cfg, Scenario(duration_s=30.0, laptop_dies_at_s=15.0))
    ev = {name: t for t, name, _ in res.phone_events}
    assert 15.0 < ev["watchdog_hover"] < 15.0 + 0.5
    assert 17.5 < ev["watchdog_handback"] < 18.5
    assert not res.trace.vs_enabled[-1]


def test_watchdog_rules():
    w = Watchdog(0.3, 3.0)
    assert w.check(0.0) == Watchdog.NONE  # disarmed until the first command
    w.feed(0.0)
    assert w.check(0.29) == Watchdog.NONE
    assert w.check(0.31) == Watchdog.HOVER
    assert w.check(0.5) == Watchdog.NONE  # fires once
    w.feed(0.6)
    assert w.check(0.95) == Watchdog.HOVER  # new silence, fires again
    assert w.check(3.61) == Watchdog.HANDBACK
    assert w.check(10.0) == Watchdog.NONE  # disarmed after hand-back
