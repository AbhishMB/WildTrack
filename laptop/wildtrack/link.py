"""Phone link: WebSocket client (telemetry and events in, commands out) and the pose ring buffer.

Video arrives later than telemetry, so every telemetry message is stamped with its
laptop arrival time and kept for a few seconds; a frame's pose is looked up at its
capture time (arrival - video lag) and interpolated between neighbouring samples.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import time
from collections import deque
from typing import Callable

from .models import Attitude, Pose, Telemetry

log = logging.getLogger(__name__)


def _lerp(a: float, b: float, k: float) -> float:
    return a + (b - a) * k


def _lerp_angle(a: float, b: float, k: float) -> float:
    d = (b - a + 180.0) % 360.0 - 180.0
    return (a + d * k + 180.0) % 360.0 - 180.0


def _lerp_att(a: Attitude, b: Attitude, k: float) -> Attitude:
    return Attitude(_lerp(a.pitch, b.pitch, k), _lerp(a.roll, b.roll, k), _lerp_angle(a.yaw, b.yaw, k))


class PoseRing:
    """Last `span_s` seconds of telemetry, by laptop arrival time. Pure."""

    def __init__(self, span_s: float):
        self.span_s = span_s
        self._t: deque[float] = deque()
        self._tel: deque[Telemetry] = deque()

    def __len__(self) -> int:
        return len(self._t)

    def push(self, t: float, tel: Telemetry) -> None:
        if self._t and t < self._t[-1]:
            return  # out of order; arrival times are monotonic so this is a duplicate
        self._t.append(t)
        self._tel.append(tel)
        while self._t and self._t[0] < t - self.span_s:
            self._t.popleft()
            self._tel.popleft()

    def latest(self) -> tuple[float, Telemetry] | None:
        return (self._t[-1], self._tel[-1]) if self._t else None

    def at(self, t: float, max_extrapolation_s: float) -> Pose | None:
        """Pose at time t, interpolated; None if t is outside the buffer by more than the tolerance."""
        if not self._t:
            return None
        ts = self._t
        if t >= ts[-1]:
            return self._tel[-1].pose(t) if t - ts[-1] <= max_extrapolation_s else None
        if t <= ts[0]:
            return self._tel[0].pose(t) if ts[0] - t <= max_extrapolation_s else None
        i = bisect.bisect_right(ts, t)
        t0, t1 = ts[i - 1], ts[i]
        a, b = self._tel[i - 1], self._tel[i]
        k = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return Pose(
            t=t,
            lat=_lerp(a.lat, b.lat, k),
            lon=_lerp(a.lon, b.lon, k),
            alt_rel=_lerp(a.alt_rel_m, b.alt_rel_m, k),
            att=_lerp_att(a.att, b.att, k),
            gimbal_att=_lerp_att(a.gimbal, b.gimbal, k),
            vel_ned=tuple(_lerp(x, y, k) for x, y in zip(a.vel_ned, b.vel_ned)),  # type: ignore[arg-type]
        )


class PhoneLink:
    """Keeps a WebSocket connection to the bridge app open and dispatches what arrives."""

    def __init__(
        self,
        url: str,
        reconnect_s: float,
        on_telemetry: Callable[[Telemetry, dict, float], None],
        on_event: Callable[[str, str, float], None],
        on_state: Callable[[bool, float], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.url = url
        self.reconnect_s = reconnect_s
        self.on_telemetry = on_telemetry
        self.on_event = on_event
        self.on_state = on_state
        self.clock = clock
        self._ws = None
        self.connected = False
        self.rx_count = 0
        self.bad_count = 0

    async def run(self) -> None:
        from websockets.asyncio.client import connect

        while True:
            try:
                async with connect(self.url, open_timeout=3, ping_interval=1, ping_timeout=2, max_queue=64) as ws:
                    self._ws = ws
                    self.connected = True
                    log.info("phone link up: %s", self.url)
                    self.on_state(True, self.clock())
                    async for raw in ws:
                        self._dispatch(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.connected:
                    log.warning("phone link lost: %s", exc)
                else:
                    log.debug("phone link connect failed: %s", exc)
            finally:
                if self.connected:
                    self.connected = False
                    self.on_state(False, self.clock())
                self._ws = None
            await asyncio.sleep(self.reconnect_s)

    def _dispatch(self, raw) -> None:
        t = self.clock()
        try:
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "telemetry":
                self.on_telemetry(Telemetry.from_msg(msg), msg, t)
                self.rx_count += 1
            elif kind == "event":
                self.on_event(str(msg.get("name")), str(msg.get("reason", "")), t)
            else:
                log.debug("ignored message type %r", kind)
        except (ValueError, KeyError, TypeError) as exc:
            self.bad_count += 1
            log.warning("bad message from phone (%s): %.200s", exc, raw)

    async def send(self, msg: dict) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(json.dumps(msg, allow_nan=False))
            return True
        except Exception as exc:
            log.warning("send failed: %s", exc)
            return False

