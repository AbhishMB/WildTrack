import dataclasses
import math

import numpy as np
import pytest

from wildtrack.controller import DroneState, FollowController
from wildtrack.models import TargetEstimate


def target(n=0.0, e=0.0, vn=0.0, ve=0.0, t=0.0):
    return TargetEstimate(t, n, e, vn, ve, np.eye(4))


def drone(n=0.0, e=0.0, t=0.0, yaw=0.0, agl=60.0, vn=0.0, ve=0.0):
    return DroneState(t, n, e, vn, ve, yaw, agl, agl)


@pytest.fixture
def ctrl(cfg):
    return FollowController(cfg.controller, cfg.safety)


def test_setpoint_is_standoff_along_current_bearing(ctrl):
    assert ctrl.setpoint(0, 0, 0, 100) == pytest.approx((0, 40))  # drone to the east: stays east
    sn, se = ctrl.setpoint(0, 0, -30, -30)
    assert math.hypot(sn, se) == pytest.approx(40)
    assert math.degrees(math.atan2(se, sn)) == pytest.approx(-135)


def test_never_directly_above(ctrl):
    sn, se = ctrl.setpoint(10, 10, 10, 10)
    assert math.hypot(sn - 10, se - 10) == pytest.approx(40)


def test_keeps_bearing_when_close_overhead(ctrl):
    ctrl.setpoint(0, 0, 0, 100)  # bearing 90
    sn, se = ctrl.setpoint(0, 0, 1.0, 0.5)  # within bearing_hold_m
    assert (sn, se) == pytest.approx((0, 40))


def test_at_setpoint_with_static_animal_commands_nothing(ctrl):
    c = ctrl.step(drone(0, 40, yaw=-90), target())
    assert math.hypot(c.vn, c.ve) == pytest.approx(0, abs=1e-9)
    assert c.vd == 0
    assert c.yaw_rate == pytest.approx(0)


def test_feedforward_matches_animal_velocity(cfg):
    ctrl = FollowController(cfg.controller, cfg.safety)
    ctrl.reset(1.0, 0.5)  # already moving with the animal
    c = ctrl.step(drone(0, 40, yaw=-90, vn=1.0, ve=0.5), target(vn=1.0, ve=0.5))
    assert (c.vn, c.ve) == pytest.approx((1.0, 0.5), abs=1e-6)


def test_speed_limit(ctrl, cfg):
    ctrl.reset()
    t = 0.0
    for _ in range(200):
        c = ctrl.step(drone(500, 500, t=t), target())
        t += 0.1
    assert math.hypot(c.vn, c.ve) == pytest.approx(cfg.controller.v_max_mps, abs=1e-6)


def test_acceleration_limit(ctrl, cfg):
    ctrl.reset()
    c1 = ctrl.step(drone(300, 0, t=0.0), target())
    c2 = ctrl.step(drone(300, 0, t=0.1), target())
    step = cfg.controller.a_max_mps2 * 0.1
    assert math.hypot(c1.vn, c1.ve) <= step + 1e-9
    assert math.hypot(c2.vn - c1.vn, c2.ve - c1.ve) <= step + 1e-9


def test_integrator_clamped_and_reset(ctrl, cfg):
    ctrl.reset()
    t = 0.0
    for _ in range(5000):  # small steady error for a long time
        ctrl.step(drone(0, 43, t=t, yaw=-90), target())
        t += 0.1
    assert math.hypot(ctrl.i_n, ctrl.i_e) <= cfg.controller.integrator_limit_mps + 1e-9
    assert math.hypot(ctrl.i_n, ctrl.i_e) > 0.5
    ctrl.reset()
    assert ctrl.i_n == ctrl.i_e == 0.0


def test_yaw_turns_toward_animal_and_is_limited(ctrl, cfg):
    c = ctrl.step(drone(0, -40, yaw=0.0), target())  # animal due east: turn clockwise
    assert c.yaw_rate > 0
    c = ctrl.step(drone(0, 40, yaw=0.0, t=0.1), target())  # animal due west
    assert c.yaw_rate == pytest.approx(-cfg.controller.yaw_rate_max_dps)
    c = ctrl.step(drone(-40, 0, yaw=10.0, t=0.2), target())  # 10 deg off
    assert c.yaw_rate == pytest.approx(-cfg.controller.yaw_kp * 10.0)


def test_gimbal_pitch_from_geometry(ctrl):
    c = ctrl.step(drone(0, 40, yaw=-90, agl=60), target())
    assert c.gimbal_pitch == pytest.approx(-math.degrees(math.atan2(60, 40)))
    below = ctrl.gimbal_pitch(drone(0, 40, agl=60), target(), box_offset_deg=10.0)
    assert below < c.gimbal_pitch  # box below centre: look further down


def test_altitude_band(ctrl, cfg):
    assert ctrl.step(drone(0, 40, agl=30.0), target()).vd < 0  # climb
    assert ctrl.step(drone(0, 40, agl=95.0, t=0.1), target()).vd > 0  # descend
    assert ctrl.step(drone(0, 40, agl=60.0, t=0.2), target()).vd == 0
    assert abs(ctrl.step(drone(0, 40, agl=0.0, t=0.3), target()).vd) <= cfg.controller.vz_max_mps


def test_hover(ctrl):
    c = ctrl.hover(1.0, -60.0)
    assert (c.vn, c.ve, c.vd, c.yaw_rate, c.gimbal_pitch) == (0, 0, 0, 0, -60.0)


def test_phase_margin_rule(cfg):
    """Spec: margin ~ 90 deg - Kp*tau; Kp*tau <= 0.4 rad leaves ~67 deg."""
    k = cfg.controller
    assert k.kp * k.tau_s <= k.max_kp_tau
    assert 90 - math.degrees(k.max_kp_tau) == pytest.approx(67, abs=0.2)


def test_derivative_acts_on_velocity_error(cfg):
    """D term: Kd times the filtered (v_animal - v_drone); zero when the drone keeps pace."""
    k = dataclasses.replace(cfg.controller, ki=0.0, kd=0.5, d_filter_s=0.0, a_max_mps2=100.0)
    on = FollowController(k, cfg.safety)
    off = FollowController(dataclasses.replace(k, kd=0.0), cfg.safety)
    for c in (on, off):
        c.reset(0.0, 0.0)
    # drone at the setpoint, animal moving north at 1 m/s, drone still at rest
    c_on = on.step(drone(0, 40, yaw=-90, vn=0.0), target(vn=1.0))
    c_off = off.step(drone(0, 40, yaw=-90, vn=0.0), target(vn=1.0))
    assert c_on.vn - c_off.vn == pytest.approx(0.5 * 1.0, abs=1e-9)
    # once the drone keeps pace with the animal the derivative term vanishes
    on.reset(1.0, 0.0)
    off.reset(1.0, 0.0)
    assert on.step(drone(0, 40, yaw=-90, vn=1.0, t=1.0), target(vn=1.0, t=1.0)).vn == pytest.approx(
        off.step(drone(0, 40, yaw=-90, vn=1.0, t=1.0), target(vn=1.0, t=1.0)).vn, abs=1e-9)


def test_derivative_has_no_kick_on_setpoint_jump(cfg):
    """The derivative uses velocities, so a jump in the standoff point adds nothing to it."""
    k = dataclasses.replace(cfg.controller, kd=0.5, d_filter_s=0.0)
    ctrl = FollowController(k, cfg.safety)
    ctrl.step(drone(0, 40, yaw=-90), target())
    ctrl.step(drone(0, 40, yaw=-90, t=0.1), target(n=30.0, t=0.1))  # target estimate jumps 30 m
    assert (ctrl.d_n, ctrl.d_e) == pytest.approx((0.0, 0.0))
