"""Video-lag calibration (docs/SPEC.md, "Timing").

Yaw the drone back and forth (on the bench: turn it by hand with the gimbal level),
then cross-correlate the image's horizontal optical flow with the yaw rate from
telemetry. The offset of the correlation peak is how far video lags telemetry.
The maths here is pure; `calibrate()` does the recording.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from pathlib import Path

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

FLOW_WIDTH = 320  # frames are downscaled to this width before phase correlation


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(a)))


def rate(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference rate, at the midpoints between samples."""
    dt = np.diff(t)
    ok = dt > 1e-6
    return (0.5 * (t[1:] + t[:-1]))[ok], (np.diff(x) / np.where(ok, dt, 1.0))[ok]


def image_yaw_rate(times: list[float], grays: list[np.ndarray], hfov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal image motion between consecutive frames -> apparent yaw rate in deg/s."""
    import cv2

    if len(grays) < 2:
        return np.array([]), np.array([])
    w = grays[0].shape[1]
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    win = cv2.createHanningWindow((grays[0].shape[1], grays[0].shape[0]), cv2.CV_32F)
    ts, rs = [], []
    for k in range(1, len(grays)):
        (dx, _dy), _ = cv2.phaseCorrelate(grays[k - 1], grays[k], win)
        dt = times[k] - times[k - 1]
        if dt <= 0:
            continue
        # turning right (yaw increasing) moves the scene left: dx < 0
        rs.append(-math.degrees(math.atan2(dx, f)) / dt)
        ts.append(0.5 * (times[k] + times[k - 1]))
    return np.array(ts), np.array(rs)


def estimate_lag(
    t_video: np.ndarray,
    r_video: np.ndarray,
    t_tel: np.ndarray,
    r_tel: np.ndarray,
    max_lag_s: float = 1.5,
    step_s: float = 0.005,
) -> tuple[float, float]:
    """Lag L maximising corr(video(t), telemetry(t - L)). Returns (L seconds, peak correlation)."""
    lo = max(t_video[0], t_tel[0] + max_lag_s)
    hi = min(t_video[-1], t_tel[-1])
    if hi - lo < 2.0:
        raise ValueError("need at least a few seconds where video and telemetry overlap")
    grid = np.arange(lo, hi, step_s)
    v = np.interp(grid, t_video, r_video)
    v = (v - v.mean()) / (v.std() + 1e-9)
    best, best_c = 0.0, -np.inf
    for lag in np.arange(0.0, max_lag_s + 1e-9, step_s):
        u = np.interp(grid - lag, t_tel, r_tel)
        u = (u - u.mean()) / (u.std() + 1e-9)
        c = float(np.mean(u * v))
        if c > best_c:
            best, best_c = float(lag), c
    return best, best_c


def write_lag(config_path: Path, lag_s: float) -> None:
    """Replace video.lag_s in config.yaml, keeping comments."""
    text = config_path.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^(\s*lag_s:\s*)[0-9.]+", lambda m: f"{m.group(1)}{lag_s:.3f}", text, count=1)
    if n != 1:
        raise ValueError("lag_s not found in config")
    config_path.write_text(new, encoding="utf-8")


async def calibrate(cfg: Config, duration_s: float) -> tuple[float, float]:
    """Record telemetry and video for duration_s while the operator yaws the drone back and forth."""
    import cv2

    from .tello import make_link
    from .video_in import FrameSlot, RtspReader

    tel_t: list[float] = []
    tel_yaw: list[float] = []
    pitches: list[float] = []

    def on_tel(tel, msg, t):
        if math.isfinite(tel.gimbal.yaw):
            tel_t.append(t)
            tel_yaw.append(tel.gimbal.yaw)
            pitches.append(tel.gimbal.pitch)

    link = make_link(cfg, on_tel, lambda *a: None, lambda *a: None)
    slot = FrameSlot()
    reader = RtspReader(cfg.video, slot)
    reader.start()
    task = asyncio.create_task(link.run())
    vid_t: list[float] = []
    grays: list[np.ndarray] = []
    print(f"Recording {duration_s:.0f} s: yaw the drone left and right, about 1 s each way, gimbal level.")
    idx = -1
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        f = await asyncio.to_thread(slot.wait_newer, idx, 0.5)
        if f is None:
            continue
        idx = f.idx
        h, w = f.image.shape[:2]
        small = cv2.resize(f.image, (FLOW_WIDTH, int(h * FLOW_WIDTH / w)), interpolation=cv2.INTER_AREA)
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
        vid_t.append(f.t_arrival)
    task.cancel()
    reader.stop()
    if pitches and np.median(pitches) < -30:
        log.warning("gimbal pitched %.0f deg: yaw shows up as image rotation, level the gimbal for a clean peak",
                    float(np.median(pitches)))
    if len(tel_t) < 20 or len(vid_t) < 20:
        raise RuntimeError(f"not enough data: {len(tel_t)} telemetry, {len(vid_t)} frames")
    tt, tr = rate(np.array(tel_t), unwrap_deg(np.array(tel_yaw)))
    vt, vr = image_yaw_rate(vid_t, grays, cfg.camera.hfov_deg)
    return estimate_lag(vt, vr, tt, tr)
