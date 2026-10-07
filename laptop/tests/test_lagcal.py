import math

import numpy as np
import pytest

from wildtrack.lagcal import estimate_lag, image_yaw_rate, rate, unwrap_deg, write_lag


def test_recovers_a_known_lag():
    rng = np.random.default_rng(0)
    lag = 0.32
    tel_t = np.arange(0.0, 20.0, 0.1) + rng.normal(0, 0.005, 200)
    yaw = 40.0 * np.sin(2 * np.pi * 0.4 * tel_t) + 40.0 * np.sin(2 * np.pi * 0.13 * tel_t)
    tt, tr = rate(tel_t, yaw)
    vid_t = np.arange(0.5, 20.0, 1 / 15.0)
    true_rate = lambda t: 40 * 2 * np.pi * 0.4 * np.cos(2 * np.pi * 0.4 * t) + 40 * 2 * np.pi * 0.13 * np.cos(2 * np.pi * 0.13 * t)  # noqa: E731
    vr = true_rate(vid_t - lag) + rng.normal(0, 5.0, len(vid_t))
    est, peak = estimate_lag(vid_t, vr, tt, tr)
    assert est == pytest.approx(lag, abs=0.02)
    assert peak > 0.8


def test_image_rate_from_shifted_texture():
    rng = np.random.default_rng(1)
    world = rng.random((180, 2000)).astype(np.float32)
    world = np.cumsum(np.cumsum(world, 0), 1)  # smooth it a bit
    world = (world - world.mean()) / world.std()
    hfov, w = 80.0, 320
    f = (w / 2) / math.tan(math.radians(hfov / 2))
    shift = 4  # pixels per frame, scene moving left = turning right
    times, grays = [], []
    for k in range(10):
        x0 = 500 + k * shift
        grays.append(np.ascontiguousarray(world[:, x0 : x0 + w]))
        times.append(k / 15.0)
    t, r = image_yaw_rate(times, grays, hfov)
    expect = math.degrees(math.atan2(shift, f)) * 15.0
    assert np.median(r) == pytest.approx(expect, rel=0.05)


def test_unwrap():
    assert list(unwrap_deg(np.array([170.0, -175.0, -160.0]))) == pytest.approx([170.0, 185.0, 200.0])


def test_write_lag_only_touches_video_lag(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("video:\n  lag_s: 0.30   # comment\nsim:\n  video_lag_s: 0.30\n", encoding="utf-8")
    write_lag(p, 0.412)
    assert p.read_text(encoding="utf-8") == "video:\n  lag_s: 0.412   # comment\nsim:\n  video_lag_s: 0.30\n"
