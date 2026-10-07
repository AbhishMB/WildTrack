import math

import numpy as np

from wildtrack.estimator import CVKalman
from wildtrack.models import Fix


def track(kf, rng, t0, t1, vn=1.5, ve=-0.5, sigma=3.0, rate=10.0):
    t = t0
    while t < t1:
        n, e = vn * t, ve * t
        kf.update(Fix(t, 0, 0, n + rng.normal(0, sigma), e + rng.normal(0, sigma), sigma))
        t += 1.0 / rate


def test_converges_on_a_walking_animal(cfg):
    """Process noise is sized for a running animal, so single velocity estimates are noisy
    (~0.6 m/s with 3 m fixes); the filter must be unbiased and consistent with its covariance."""
    kf = CVKalman(cfg.estimator)
    rng = np.random.default_rng(0)
    track(kf, rng, 0.0, 10.0)
    vels, inside = [], 0
    t = 10.0
    for _ in range(300):
        track(kf, rng, t, t + 0.1)
        t += 0.1
        e = kf.estimate()
        vels.append((e.vn, e.ve))
        err = np.array([e.n - 1.5 * e.t, e.e + 0.5 * e.t, e.vn - 1.5, e.ve + 0.5])
        inside += float(err @ np.linalg.solve(e.cov, err)) < 13.28  # chi-square 4 DOF, 99%
    mean_v = np.mean(vels, axis=0)
    assert math.hypot(mean_v[0] - 1.5, mean_v[1] + 0.5) < 0.2
    assert inside / 300 > 0.95


def test_predicts_through_an_occlusion(cfg):
    kf = CVKalman(cfg.estimator)
    track(kf, np.random.default_rng(1), 0.0, 20.0)
    gap_end = kf.t + 3.0
    p = kf.peek(gap_end)
    assert math.hypot(p.n - 1.5 * gap_end, p.e + 0.5 * gap_end) < 4.0
    assert p.sigma_m > kf.estimate().sigma_m  # less sure after the gap


def test_gate(cfg):
    kf = CVKalman(cfg.estimator)
    track(kf, np.random.default_rng(2), 0.0, 10.0)
    t = kf.t + 0.5
    near = Fix(t, 0, 0, 1.5 * t, -0.5 * t, 3.0)
    far = Fix(t, 0, 0, 1.5 * t + 80, -0.5 * t, 3.0)
    assert kf.mahalanobis(near) < cfg.lock.gate_mahalanobis
    assert kf.mahalanobis(far) > cfg.lock.gate_mahalanobis


def test_ignores_out_of_order_fixes(cfg):
    kf = CVKalman(cfg.estimator)
    kf.update(Fix(10.0, 0, 0, 0.0, 0.0, 3.0))
    assert not kf.update(Fix(9.0, 0, 0, 50.0, 0.0, 3.0))
    assert kf.estimate().n == 0.0


def test_covariance_stays_symmetric_positive(cfg):
    kf = CVKalman(cfg.estimator)
    track(kf, np.random.default_rng(3), 0.0, 60.0, rate=30.0)
    P = kf.P
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) > 0)
