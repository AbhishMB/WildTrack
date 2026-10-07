"""One test per row of the spec's mode table."""

import dataclasses

import pytest

from wildtrack.fsm import Act, Ev, Mode, ModeMachine, World

GOOD = World(
    target_lockable=True,
    auto_lock_enabled=False,
    gps_sats=17,
    battery_pct=80,
    agl_m=60,
    dist_home_m=200,
    link_alive=True,
    has_fix=True,
)


def machine(cfg, mode=Mode.MANUAL):
    m = ModeMachine(cfg.safety)
    m.mode = mode
    return m


def test_manual_lock_on_target_species(cfg):
    out = machine(cfg).handle(Ev.LOCK, GOOD)
    assert out.accepted and out.dst == Mode.LOCKED and Act.FIX_TARGET in out.actions


def test_manual_lock_refused_on_non_target(cfg):
    m = machine(cfg)
    out = m.handle(Ev.LOCK, dataclasses.replace(GOOD, target_lockable=False))
    assert not out.accepted and m.mode == Mode.MANUAL and out.refusals


def test_auto_lock_needs_the_switch(cfg):
    assert not machine(cfg).handle(Ev.AUTO_LOCK, GOOD).accepted
    out = machine(cfg).handle(Ev.AUTO_LOCK, dataclasses.replace(GOOD, auto_lock_enabled=True))
    assert out.accepted and out.dst == Mode.LOCKED


@pytest.mark.parametrize("ev", [Ev.RELEASE, Ev.TRACK_LOST])
def test_locked_back_to_manual(cfg, ev):
    out = machine(cfg, Mode.LOCKED).handle(ev)
    assert out.dst == Mode.MANUAL and Act.CLEAR_TARGET in out.actions and Act.VS_DISABLE not in out.actions


def test_engage(cfg):
    out = machine(cfg, Mode.LOCKED).handle(Ev.ENGAGE, GOOD)
    assert out.accepted and out.dst == Mode.AUTO
    assert out.actions == (Act.VS_ENABLE, Act.RESET_INTEGRATOR, Act.START_COMMANDS)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"gps_sats": 9}, "GPS"),
        ({"battery_pct": 34}, "battery"),
        ({"agl_m": 39.9}, "height"),
        ({"agl_m": 80.1}, "height"),
        ({"agl_m": None}, "height"),
        ({"dist_home_m": 1001}, "radius"),
        ({"dist_home_m": None}, "home"),
        ({"link_alive": False}, "link"),
        ({"has_fix": False}, "fix"),
    ],
)
def test_engage_preconditions(cfg, change, reason):
    m = machine(cfg, Mode.LOCKED)
    out = m.handle(Ev.ENGAGE, dataclasses.replace(GOOD, **change))
    assert not out.accepted and m.mode == Mode.LOCKED
    assert any(reason in r for r in out.refusals)


@pytest.mark.parametrize("agl", [40.0, 80.0])
def test_altitude_band_is_inclusive(cfg, agl):
    assert machine(cfg, Mode.LOCKED).handle(Ev.ENGAGE, dataclasses.replace(GOOD, agl_m=agl)).accepted


def test_engage_only_from_locked(cfg):
    assert not machine(cfg, Mode.MANUAL).handle(Ev.ENGAGE, GOOD).accepted


def test_no_fix_hovers(cfg):
    out = machine(cfg, Mode.AUTO).handle(Ev.NO_FIX)
    assert out.dst == Mode.LOST and out.actions == (Act.HOVER,)


def test_reacquire_resumes_with_fresh_integrator(cfg):
    out = machine(cfg, Mode.LOST).handle(Ev.REACQUIRED)
    assert out.dst == Mode.AUTO and Act.RESET_INTEGRATOR in out.actions


def test_lost_timeout_hands_back(cfg):
    out = machine(cfg, Mode.LOST).handle(Ev.LOST_TIMEOUT)
    assert out.dst == Mode.MANUAL and Act.VS_DISABLE in out.actions and Act.ALERT_PILOT in out.actions


@pytest.mark.parametrize("mode", [Mode.AUTO, Mode.LOST])
def test_dji_ends_virtual_stick(cfg, mode):
    out = machine(cfg, mode).handle(Ev.VS_EXIT)
    assert out.dst == Mode.MANUAL
    assert Act.STOP_COMMANDS in out.actions and Act.ALERT_PILOT in out.actions
    assert Act.VS_ENABLE not in out.actions  # never re-enable on our own


@pytest.mark.parametrize("mode", [Mode.AUTO, Mode.LOST])
@pytest.mark.parametrize("ev", [Ev.RELEASE, Ev.OUT_OF_RADIUS, Ev.LOW_BATTERY, Ev.LAPTOP_SILENT, Ev.LINK_LOST])
def test_hand_back_turns_virtual_stick_off(cfg, mode, ev):
    out = machine(cfg, mode).handle(ev)
    assert out.dst == Mode.MANUAL and Act.VS_DISABLE in out.actions and Act.STOP_COMMANDS in out.actions


def test_no_transition_ever_leads_back_into_auto_except_engage_and_reacquire(cfg):
    from wildtrack.fsm import TABLE

    into_auto = {ev for row in TABLE if row.dst == Mode.AUTO for ev in row.events}
    assert into_auto == {Ev.ENGAGE, Ev.REACQUIRED}
    assert all(row.src == {Mode.LOST} for row in TABLE if Ev.REACQUIRED in row.events)
