"""A Tello stand-in speaking the real UDP protocol (commands in, state packets out).

Used by the tests and for trying the laptop app without hardware. No video: the live
simulator (sim/live.py) covers the video and detection path.

Kinematics follow the SDK's rc semantics: rc (right, forward, up, yaw) in -100..100 is a
body-frame velocity; position is kept in the takeoff frame (x forward, y right at takeoff).
"""

from __future__ import annotations

import asyncio
import math
import socket
import time

from ..config import TelloCfg


class FakeTello(asyncio.DatagramProtocol):
    def __init__(self, t: TelloCfg, host: str = "127.0.0.1", state_hz: float = 10.0, battery: float = 90.0):
        self.t = t
        self.host = host
        self.state_hz = state_hz
        self.battery = battery
        self.transport: asyncio.DatagramTransport | None = None
        self.peer: tuple[str, int] | None = None
        self.log: list[str] = []  # every datagram received, in order
        self.sdk_mode = False
        self.streaming = False
        self.flying = False
        self.emergency = False
        self.rc = (0, 0, 0, 0)
        self.x = self.y = 0.0  # takeoff frame, metres
        self.h = 0.0  # metres
        self.yaw = 0.0
        self.vx = self.vy = 0.0
        self._task: asyncio.Task | None = None
        self.rc_count = 0

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(lambda: self, local_addr=(self.host, self.t.cmd_port))
        # State goes out on its own socket: on Windows, sending to a closed port (the laptop mid-retry)
        # raises an ICMP error that would otherwise kill the command socket's receive side.
        self.state_tx, _ = await loop.create_datagram_endpoint(asyncio.DatagramProtocol, family=socket.AF_INET)
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
        for tr in (self.transport, getattr(self, "state_tx", None)):
            if tr:
                tr.close()

    # ---- protocol

    def datagram_received(self, data: bytes, addr) -> None:
        text = data.decode().strip()
        self.peer = addr
        if text.startswith("rc "):
            self.rc_count += 1
            if self.flying and not self.emergency:
                self.rc = tuple(int(float(v)) for v in text.split()[1:5])  # type: ignore[assignment]
            return
        self.log.append(text)
        reply = "ok"
        if text == "command":
            self.sdk_mode = True
        elif not self.sdk_mode:
            reply = "error"
        elif text == "streamon":
            self.streaming = True
        elif text == "takeoff":
            if self.battery < 10 or self.flying:
                reply = "error"
            else:
                self.flying, self.emergency, self.h = True, False, 1.0
        elif text == "land":
            self.flying, self.rc, self.h = False, (0, 0, 0, 0), 0.0
        elif text == "emergency":
            self.flying, self.emergency, self.rc, self.h = False, True, (0, 0, 0, 0), 0.0
            return
        else:
            reply = "error"
        if self.transport and self.peer:
            self.transport.sendto(reply.encode(), self.peer)

    # ---- physics and state

    async def _run(self) -> None:
        since_state = 0.0
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            dt, last = now - last, now  # real elapsed time: Windows timers sleep longer than asked
            right, fwd, up, yawr = self.rc
            k = self.t.rc_per_mps
            vf, vr = fwd / k, right / k
            psi = math.radians(self.yaw)
            self.vx = vf * math.cos(psi) - vr * math.sin(psi)
            self.vy = vf * math.sin(psi) + vr * math.cos(psi)
            self.x += self.vx * dt
            self.y += self.vy * dt
            if self.flying:
                self.h = max(0.1, self.h + up / k * dt)
                self.yaw = (self.yaw + yawr * self.t.yaw_dps_per_unit * dt + 180.0) % 360.0 - 180.0
            since_state += dt
            if since_state >= 1.0 / self.state_hz:
                since_state = 0.0
                self._send_state()

    def state_text(self) -> str:
        u = self.t.state_speed_mps
        vgx = self.vx / u * self.t.vg_forward_sign
        # vgy follows vg_right_sign; the takeoff frame's y is to the right
        vgy = self.vy / u * self.t.vg_right_sign
        return (f"pitch:0;roll:0;yaw:{int(round(self.yaw))};vgx:{int(round(vgx))};vgy:{int(round(vgy))};vgz:0;"
                f"templ:60;temph:62;tof:{int(self.h * 100)};h:{int(self.h * 100)};bat:{int(self.battery)};"
                f"baro:0.0;time:0;agx:0.0;agy:0.0;agz:-998.0;\r\n")

    def _send_state(self) -> None:
        if getattr(self, "state_tx", None) is not None:
            self.state_tx.sendto(self.state_text().encode(), (self.host, self.t.state_port))
