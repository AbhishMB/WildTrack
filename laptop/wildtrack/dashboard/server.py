"""Ground-station web app: FastAPI + WebSocket, served by the laptop to any browser on the field Wi-Fi.

Each browser gets its own send queue, so a slow phone never stalls the command loop.
Only hosts in dashboard.control_hosts may LOCK and ENGAGE; RELEASE works from every
client, because handing control back to the pilot must always be possible.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import DashboardCfg

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"

# action, payload, client may control -> reply {ok, message}
CommandFn = Callable[[str, dict, bool], Awaitable[dict | None]]


class TileStore:
    """Offline map tiles from an XYZ directory ({z}/{x}/{y}.png|jpg) or one MBTiles file."""

    def __init__(self, tiles_dir: Path | None, mbtiles: Path | None):
        self.dir = tiles_dir if tiles_dir and tiles_dir.is_dir() else None
        self.db: sqlite3.Connection | None = None
        self.mb_type = "image/png"
        self._lock = threading.Lock()
        if mbtiles and mbtiles.is_file():
            self.db = sqlite3.connect(str(mbtiles), check_same_thread=False)
            row = self.db.execute("SELECT value FROM metadata WHERE name='format'").fetchone()
            if row and row[0] in ("jpg", "jpeg"):
                self.mb_type = "image/jpeg"
        if self.dir is None and self.db is None:
            log.warning("no offline map tiles (dashboard.tiles_dir / dashboard.mbtiles): the map will be blank")

    def get(self, z: int, x: int, y: int) -> tuple[bytes, str] | None:
        if self.db is not None:
            with self._lock:
                row = self.db.execute(
                    "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                    (z, x, (1 << z) - 1 - y),
                ).fetchone()
            if row:
                return bytes(row[0]), self.mb_type
        if self.dir is not None:
            for ext, mt in ((".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".webp", "image/webp")):
                p = self.dir / str(z) / str(x) / f"{y}{ext}"
                if p.is_file():
                    return p.read_bytes(), mt
        return None


class Client:
    def __init__(self, ws: WebSocket, control: bool):
        self.ws = ws
        self.control = control
        self.msgs: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        self.frame: tuple[str, bytes] | None = None
        self.wake = asyncio.Event()

    def send_json(self, msg: dict) -> None:
        text = json.dumps(msg, allow_nan=False, default=str)
        if self.msgs.full():
            try:
                self.msgs.get_nowait()  # drop the oldest; state messages supersede each other
            except asyncio.QueueEmpty:
                pass
        self.msgs.put_nowait(text)
        self.wake.set()

    def send_frame(self, meta: str, jpeg: bytes) -> None:
        self.frame = (meta, jpeg)  # newest frame only
        self.wake.set()

    async def pump(self) -> None:
        while True:
            await self.wake.wait()
            self.wake.clear()
            while not self.msgs.empty():
                await self.ws.send_text(self.msgs.get_nowait())
            if self.frame is not None:
                meta, jpeg = self.frame
                self.frame = None
                await self.ws.send_text(meta)
                await self.ws.send_bytes(jpeg)


class Hub:
    def __init__(self, cfg: DashboardCfg, ui_config: dict):
        self.cfg = cfg
        self.ui_config = ui_config
        self.clients: set[Client] = set()
        self.on_command: CommandFn | None = None
        self.last: dict[str, dict] = {}

    def publish(self, msg: dict) -> None:
        """Send to every browser. Must be called on the event loop."""
        kind = msg.get("type", "")
        if kind in ("state", "animal"):
            self.last[kind] = msg
        for c in list(self.clients):
            c.send_json(msg)

    def publish_frame(self, meta: dict, jpeg: bytes) -> None:
        text = json.dumps(meta, allow_nan=False, default=str)
        for c in list(self.clients):
            c.send_frame(text, jpeg)

    def is_control_host(self, host: str | None) -> bool:
        return host in self.cfg.control_hosts

    async def serve(self, ws: WebSocket) -> None:
        await ws.accept()
        host = ws.client.host if ws.client else None
        client = Client(ws, self.is_control_host(host))
        self.clients.add(client)
        client.send_json({"type": "hello", "control": client.control, "config": self.ui_config})
        for m in self.last.values():
            client.send_json(m)
        pump = asyncio.create_task(client.pump())
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") != "cmd":
                    continue
                action = str(msg.get("action", ""))
                if self.on_command is None:
                    reply = {"ok": False, "message": "not ready"}
                else:
                    reply = await self.on_command(action, msg, client.control)
                if reply is not None:
                    client.send_json({"type": "reply", "action": action, **reply})
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            log.exception("dashboard client error")
        finally:
            pump.cancel()
            self.clients.discard(client)


def create_app(
    hub: Hub,
    tiles: TileStore,
    export: Callable[[str], tuple[str, str]] | None = None,
) -> FastAPI:
    """export(kind) -> (text, filename) for kind in {"gpx", "kml"}."""
    app = FastAPI(title="wildtrack ground station", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/tiles/{z}/{x}/{y}.png")
    def tile(z: int, x: int, y: int) -> Response:
        hit = tiles.get(z, x, y)
        if hit is None:
            raise HTTPException(404)
        data, mt = hit
        return Response(data, media_type=mt, headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/qr.svg")
    def qr(text: str = Query(..., max_length=500)) -> Response:
        import segno

        buf = io.BytesIO()
        segno.make(text, error="m").save(buf, kind="svg", scale=6, border=2, dark="#000", light="#fff")
        return Response(buf.getvalue(), media_type="image/svg+xml")

    @app.get("/api/export.{kind}")
    def export_track(kind: str) -> Response:
        if export is None or kind not in ("gpx", "kml"):
            raise HTTPException(404)
        text, name = export(kind)
        mt = "application/gpx+xml" if kind == "gpx" else "application/vnd.google-earth.kml+xml"
        return Response(text, media_type=mt, headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.serve(ws)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def ui_config(cfg: DashboardCfg, extra: dict[str, Any] | None = None) -> dict:
    out = {
        "stale_pin_s": cfg.stale_pin_s,
        "trail_minutes": cfg.trail_minutes,
        "map_center": list(cfg.map_center),
        "map_zoom": cfg.map_zoom,
        "map_max_zoom": cfg.map_max_zoom,
    }
    out.update(extra or {})
    return out
