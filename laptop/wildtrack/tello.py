"""Direct Wi-Fi link to a DJI/Ryze Tello: replaces the phone relay.

The laptop joins the Tello's own Wi-Fi network and talks UDP (Tello SDK 2.0):
    commands  laptop -> 192.168.10.1:8889, replies "ok"/"error" to the same socket
    state     drone  -> laptop:8890, ~10 Hz text "pitch:0;roll:0;yaw:0;vgx:0;...;h:0;bat:87;..."
    video     drone  -> laptop:11111, H.264 (after `streamon`), read by video_in.py

TelloLink has the same interface as PhoneLink and speaks the same wire format to the rest of
the app: it turns state packets into the telemetry message and turns `vel`/`vs_enable`/
`vs_disable` messages into `rc` sticks. The Tello has no GPS and no gimbal, so:
  * position is dead-reckoned from its velocity reports, in metres from the takeoff point,
    and reported as lat/lon around tello.home;
  * the camera is fixed to the airframe, so the "gimbal" attitude is the aircraft's attitude
    plus the camera mount angle, and gimbal commands are ignored.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Callable

from .config import Config, TelloCfg
from .geo import ned_to_latlon, wrap180
from .models import LatLon, Telemetry
from .watchdog import Watchdog

log = logging.getLogger(__name__)

STATE_KEYS = ("pitch", "roll", "yaw", "vgx", "vgy", "vgz", "h", "bat", "tof", "baro", "templ", "temph", "time")


# ----------------------------------------------------------------------------- pure helpers


def parse_state(text: str) -> dict[str, float]:
    """'pitch:0;roll:-1;yaw:12;...;' -> {'pitch': 0.0, ...}. Unknown or broken fields are skipped."""
    out: dict[str, float] = {}
    for part in text.strip().split(";"):
        key, sep, val = part.partition(":")
        if not sep:
            continue
        try:
            out[key.strip()] = float(val)
        except ValueError:
            continue
    return out


def clip_rc(x: float) -> int:
    return max(-100, min(100, int(round(x))))


def ground_to_rc(vn: float, ve: float, vd: float, yaw_rate_dps: float, heading_deg: float, t: TelloCfg) -> tuple[int, int, int, int]:
    """Ground-frame velocity (north, east, down) and yaw rate -> rc (right, forward, up, yaw), each -100..100.

    Pure and unit-tested: this is the Tello's equivalent of the DJI roll/pitch axis mapping.
    """
    h = math.radians(heading_deg)
    forward = vn * math.cos(h) + ve * math.sin(h)
    right = -vn * math.sin(h) + ve * math.cos(h)
    k = t.rc_per_mps
    return clip_rc(right * k), clip_rc(forward * k), clip_rc(-vd * k), clip_rc(yaw_rate_dps / t.yaw_dps_per_unit)


class DeadReckoner:
    """Integrates the Tello's velocity (takeoff frame) into north/east metres from the takeoff point."""

    def __init__(self, t: TelloCfg):
        self.t = t
        self.n = 0.0
        self.e = 0.0
        self.vn = 0.0
        self.ve = 0.0
        self._last: float | None = None

    def reset(self) -> None:
        self.n = self.e = self.vn = self.ve = 0.0
        self._last = None

    def update(self, state: dict[str, float], now: float) -> None:
        fwd = state.get("vgx", 0.0) * self.t.state_speed_mps * self.t.vg_forward_sign
        right = state.get("vgy", 0.0) * self.t.state_speed_mps * self.t.vg_right_sign
        h0 = math.radians(self.t.takeoff_heading_deg)
        self.vn = fwd * math.cos(h0) - right * math.sin(h0)
        self.ve = fwd * math.sin(h0) + right * math.cos(h0)
        if self._last is not None:
            dt = min(now - self._last, 0.5)  # a long gap is a dropout: don't integrate across it
            self.n += self.vn * dt
            self.e += self.ve * dt
        self._last = now


def telemetry_message(state: dict[str, float], dr: DeadReckoner, cfg: Config, vs_enabled: bool, seq: int, phone_ms: int) -> dict:
    """The spec's telemetry wire message, built from one Tello state packet."""
    t = cfg.tello
    home = LatLon(*t.home)
    p = ned_to_latlon(dr.n, dr.e, home, cfg.geo.earth_radius_m)
    yaw_state = state.get("yaw", 0.0)
    heading = wrap180(yaw_state + t.takeoff_heading_deg)
    pitch = state.get("pitch", 0.0)
    return {
        "type": "telemetry",
        "seq": seq,
        "t_phone_ms": phone_ms,
        "lat": p.lat,
        "lon": p.lon,
        "alt_rel_m": state.get("h", 0.0) / 100.0,
        "att": {"pitch": pitch, "roll": state.get("roll", 0.0), "yaw": heading},
        "vel_ned": {"n": dr.vn, "e": dr.ve, "d": None},
        "gimbal": {"pitch": t.camera_pitch_deg + pitch, "roll": 0.0, "yaw": heading},
        "battery_pct": state.get("bat", 0.0),
        "gps_sats": 0,
        "vs_enabled": vs_enabled,
        "flight_mode": "P",
        "home": {"lat": home.lat, "lon": home.lon},
    }


# ----------------------------------------------------------------------------- UDP plumbing


class _Replies(asyncio.DatagramProtocol):
    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    def datagram_received(self, data: bytes, addr) -> None:
        self.queue.put_nowait(data.decode("utf-8", "replace").strip())


class _State(asyncio.DatagramProtocol):
    def __init__(self, on_state: Callable[[dict[str, float]], None]) -> None:
        self.on_state = on_state

    def datagram_received(self, data: bytes, addr) -> None:
        st = parse_state(data.decode("utf-8", "replace"))
        if st:
            self.on_state(st)


class TelloLink:
    """Same interface as PhoneLink (run / send / connected), see link.py."""

    def __init__(
        self,
        cfg: Config,
        on_telemetry: Callable[[Telemetry, dict, float], None],
        on_event: Callable[[str, str, float], None],
        on_state: Callable[[bool, float], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.t = cfg.tello
        self.on_telemetry, self.on_event, self.on_state, self.clock = on_telemetry, on_event, on_state, clock
        self.connected = False
        self.rx_count = 0
        self.bad_count = 0
        self.vs_enabled = False
        self.flying = False
        self.dr = DeadReckoner(self.t)
        self.watchdog = Watchdog(int(self.t.watchdog_hover_s * 1000), int(self.t.watchdog_handback_s * 1000))
        self._rc = (0, 0, 0, 0)
        self._rc_t = -math.inf
        self._seq = 0
        self._last_state_t = -math.inf
        self._cmd_lock = asyncio.Lock()
        self._cmd: asyncio.DatagramTransport | None = None
        self._replies: _Replies | None = None
        self.last_state: dict[str, float] = {}

    # ---------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            state_tr = None
            try:
                self._replies = _Replies()
                self._cmd, _ = await loop.create_datagram_endpoint(
                    lambda: self._replies, remote_addr=(self.t.host, self.t.cmd_port),
                    local_addr=("0.0.0.0", self.t.local_cmd_port) if self.t.local_cmd_port else None,
                )
                state_tr, _ = await loop.create_datagram_endpoint(
                    lambda: _State(self._on_state_packet), local_addr=("0.0.0.0", self.t.state_port)
                )
                await self._handshake()
                self._last_state_t = self.clock()  # the timeout clock starts now, not at -infinity
                self._mark(True)
                await self._serve()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.connected:
                    log.warning("tello link lost: %s", exc)
                else:
                    log.debug("tello connect failed: %s", exc)
            finally:
                if self.connected:
                    self._mark(False)
                self.vs_enabled = False
                self.watchdog.disarm()
                for tr in (self._cmd, state_tr):
                    if tr is not None:
                        tr.close()
                self._cmd = None
            await asyncio.sleep(self.t.connect_retry_s)

    def _mark(self, up: bool) -> None:
        self.connected = up
        if up:
            self.dr.reset()
            log.info("tello link up: %s", self.t.host)
        self.on_state(up, self.clock())

    async def _handshake(self) -> None:
        # One try per socket: on Windows an unanswered send (ICMP "port unreachable") leaves the
        # socket unable to hear later replies, so a retry must start from fresh sockets (run()).
        if await self._command("command", 0.5) != "ok":
            raise TimeoutError("the Tello does not answer: is the laptop on its Wi-Fi?")
        reply = await self._command("streamon", 2.0)
        if reply != "ok":
            raise RuntimeError(f"streamon refused: {reply!r}")

    async def _serve(self) -> None:
        period = 1.0 / self.t.rc_hz
        while True:
            now = self.clock()
            if now - self._last_state_t > self.t.state_timeout_s:
                raise TimeoutError("no state packets from the Tello")
            self._watchdog_step(now)
            self._send_raw(self._current_rc(now))
            await asyncio.sleep(period)

    # ---------------------------------------------------------------- sending

    def _send_raw(self, text: str) -> None:
        if self._cmd is not None:
            self._cmd.sendto(text.encode())

    async def _command(self, text: str, timeout: float) -> str | None:
        """Send a command that gets a reply and wait for it (one at a time)."""
        assert self._replies is not None
        async with self._cmd_lock:
            while not self._replies.queue.empty():
                self._replies.queue.get_nowait()  # stale replies
            self._send_raw(text)
            try:
                return await asyncio.wait_for(self._replies.queue.get(), timeout)
            except asyncio.TimeoutError:
                return None

    def _current_rc(self, now: float) -> str:
        a, b, c, d = self._rc
        if now - self._rc_t > self.t.rc_stale_s:
            a = b = c = d = 0  # stale stick or stale velocity command: hover
        return f"rc {a} {b} {c} {d}"

    def _watchdog_step(self, now: float) -> None:
        if not self.vs_enabled:
            return
        w = self.watchdog.check(int(now * 1000))
        if w == Watchdog.HOVER:
            self.on_event("watchdog_hover", "", now)
        elif w == Watchdog.HANDBACK:
            self.vs_enabled = False
            self._rc = (0, 0, 0, 0)
            self.on_event("watchdog_handback", "", now)
            self.on_event("vs_exit", "WATCHDOG", now)

    def _on_state_packet(self, st: dict[str, float]) -> None:
        now = self.clock()
        self._last_state_t = now
        self.last_state = st
        self.flying = st.get("h", 0.0) > 5.0  # cm above the takeoff point
        self.dr.update(st, now)
        self._seq += 1
        msg = telemetry_message(st, self.dr, self.cfg, self.vs_enabled, self._seq, int(time.time() * 1000))
        self.rx_count += 1
        if self.connected:
            self.on_telemetry(Telemetry.from_msg(msg), msg, now)

    async def send(self, msg: dict) -> bool:
        """Accepts the app's wire messages plus manual-flight messages (rc, takeoff, land, emergency)."""
        if not self.connected:
            return False
        kind = msg.get("type")
        now = self.clock()
        t = self.t
        if kind == "vel":
            if not self.vs_enabled:
                return True
            lim = t.max_speed_mps
            vn, ve = float(msg["vn"]), float(msg["ve"])
            h = math.hypot(vn, ve)
            if h > lim:
                vn, ve = vn * lim / h, ve * lim / h
            heading = wrap180(self.last_state.get("yaw", 0.0) + t.takeoff_heading_deg)
            self._rc = ground_to_rc(vn, ve, float(msg.get("vd", 0.0)), float(msg.get("yaw_rate_dps", 0.0)), heading, t)
            self._rc_t = now
            self.watchdog.feed(int(now * 1000))
        elif kind == "gimbal":
            pass  # the camera is fixed to the airframe
        elif kind == "vs_enable":
            self.vs_enabled = True
            self._rc = (0, 0, 0, 0)
            self.watchdog.feed(int(now * 1000))
            self.on_event("vs_enabled", "", now)
        elif kind == "vs_disable":
            if self.vs_enabled:
                self.vs_enabled = False
                self.watchdog.disarm()
                self._rc = (0, 0, 0, 0)
                self.on_event("vs_exit", "LAPTOP_REQUEST", now)
        elif kind == "rc":  # manual sticks from the dashboard, fractions -1..1
            if self.vs_enabled:
                return False
            k = 100.0 * t.manual_speed_frac
            self._rc = (clip_rc(msg.get("right", 0.0) * k), clip_rc(msg.get("fwd", 0.0) * k),
                        clip_rc(msg.get("up", 0.0) * k), clip_rc(msg.get("yaw", 0.0) * 100.0 * t.manual_speed_frac))
            self._rc_t = now
        elif kind in ("takeoff", "land"):
            if kind == "land":
                self.vs_enabled = False
                self._rc = (0, 0, 0, 0)
            reply = await self._command(kind, 10.0)
            if reply != "ok":
                self.on_event(f"{kind}_failed", reply or "no reply", now)
                return False
        elif kind == "emergency":
            self.vs_enabled = False
            self._rc = (0, 0, 0, 0)
            for _ in range(3):  # UDP: say it more than once
                self._send_raw("emergency")
                await asyncio.sleep(0.02)
        else:
            return False
        return True


def make_link(cfg: Config, on_telemetry, on_event, on_state):
    """The link for cfg.link.backend."""
    if cfg.link.backend == "tello":
        return TelloLink(cfg, on_telemetry, on_event, on_state)
    from .link import PhoneLink

    return PhoneLink(cfg.link.ws_url, cfg.link.reconnect_s, on_telemetry, on_event, on_state)
