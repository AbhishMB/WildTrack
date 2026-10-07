"""Constant-velocity Kalman filter for the animal on the ground (docs/SPEC.md, "Estimator").

State x = [n, e, vn, ve] in metres and m/s from the home point. Each geolocation Fix
measures [n, e] with variance sigma_m^2. Between fixes the filter predicts, which
carries the target through short occlusions. Pure: no I/O.
"""

from __future__ import annotations

import math

import numpy as np

from .config import EstimatorCfg
from .models import Fix, TargetEstimate

_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def _transition(dt: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _process_noise(dt: float, accel_sigma: float) -> np.ndarray:
    """White-noise acceleration, sigma_a in m/s^2 (discrete Wiener process acceleration)."""
    q4, q3, q2 = dt**4 / 4.0, dt**3 / 2.0, dt**2
    return accel_sigma**2 * np.array(
        [[q4, 0.0, q3, 0.0], [0.0, q4, 0.0, q3], [q3, 0.0, q2, 0.0], [0.0, q3, 0.0, q2]]
    )


class CVKalman:
    def __init__(self, cfg: EstimatorCfg):
        self.cfg = cfg
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.t: float | None = None

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def reset(self) -> None:
        self.x = self.P = self.t = None

    def init(self, fix: Fix) -> None:
        self.x = np.array([fix.n, fix.e, 0.0, 0.0])
        sv = self.cfg.init_vel_sigma
        self.P = np.diag([fix.sigma_m**2, fix.sigma_m**2, sv**2, sv**2])
        self.t = fix.t

    def _predicted(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.x is not None and self.P is not None and self.t is not None
        dt = max(0.0, t - self.t)
        F = _transition(dt)
        return F @ self.x, F @ self.P @ F.T + _process_noise(dt, self.cfg.accel_sigma)

    def peek(self, t: float) -> TargetEstimate:
        """The estimate extrapolated to time t, without changing the filter."""
        x, P = self._predicted(t)
        return TargetEstimate(t, float(x[0]), float(x[1]), float(x[2]), float(x[3]), P)

    def mahalanobis(self, fix: Fix) -> float:
        """Distance of a candidate fix from the prediction at its time, in sigmas (2 DOF)."""
        x, P = self._predicted(fix.t)
        y = np.array([fix.n, fix.e]) - _H @ x
        S = _H @ P @ _H.T + np.eye(2) * fix.sigma_m**2
        return float(math.sqrt(max(0.0, y @ np.linalg.solve(S, y))))

    def update(self, fix: Fix) -> bool:
        """Fold in a fix. Returns False (and ignores it) if it is older than the state."""
        if not self.initialized:
            self.init(fix)
            return True
        assert self.t is not None
        if fix.t < self.t:
            return False
        x, P = self._predicted(fix.t)
        R = np.eye(2) * fix.sigma_m**2
        y = np.array([fix.n, fix.e]) - _H @ x
        S = _H @ P @ _H.T + R
        K = P @ _H.T @ np.linalg.inv(S)
        self.x = x + K @ y
        # Joseph form keeps P symmetric and positive definite
        I_KH = np.eye(4) - K @ _H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T
        self.t = fix.t
        return True

    def estimate(self) -> TargetEstimate:
        assert self.t is not None
        return self.peek(self.t)
