import asyncio
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from wildtrack.config import override
from wildtrack.detect import nms, tile_origins, touches_inner_edge
from wildtrack.export import to_gpx, to_kml
from wildtrack.models import Fix

FIXES = [Fix(100.0 + k, 12.29 + k * 1e-5, 76.64 + k * 1e-5, 0, 0, 4.0 + k) for k in range(5)]


def test_gpx_is_valid():
    root = ET.fromstring(to_gpx(FIXES, 1.7e9, "sortie <1>"))
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    pts = root.findall(".//g:trkpt", ns)
    assert len(pts) == 5 and pts[0].get("lat") == "12.2900000"
    assert pts[0].find("g:time", ns).text.startswith("2023-11-14T22:15:")


def test_kml_is_valid():
    root = ET.fromstring(to_kml(FIXES, 1.7e9, "sortie & co"))
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    coords = root.find(".//k:LineString/k:coordinates", ns).text.split()
    assert len(coords) == 5 and coords[0] == "76.6400000,12.2900000,0"


def test_tiles_cover_the_frame():
    for w, h in [(1280, 720), (1920, 1080), (3840, 2160), (640, 480)]:
        tiles = tile_origins(w, h, 1280, 0.2)
        covered = np.zeros((h, w), bool)
        for x, y in tiles:
            covered[y : y + 1280, x : x + 1280] = True
        assert covered.all()


def test_tile_edge_boxes_are_dropped():
    b = np.array([[0.5, 10, 50, 60, 0.9, 0], [100, 100, 200, 200, 0.9, 0], [600, 10, 640, 60, 0.9, 0]])
    # tile at x=640 in a 1920-wide frame: its left and right edges are inside the image
    assert list(touches_inner_edge(b, 640, 0, 640, 1920, 640)) == [True, False, True]
    # tile at the image's left edge: a box touching x=0 is fine
    assert list(touches_inner_edge(b, 0, 0, 640, 1920, 640)) == [False, False, True]


def test_nms_is_per_class():
    b = np.array([[0, 0, 10, 10, 0.9, 0], [1, 1, 10, 10, 0.8, 0], [0, 0, 10, 10, 0.7, 1]])
    kept = nms(b, 0.5)
    assert len(kept) == 2 and set(kept[:, 5]) == {0, 1}


# ------------------------------------------------------------------ dashboard


@pytest.fixture
def app_obj(cfg, tmp_path):
    from wildtrack.app import App
    from wildtrack.video_in import FrameSlot, StreamStats

    c = override(cfg, "logging.dir", str(tmp_path / "logs"))
    c = override(c, "geo.dem_dir", str(tmp_path / "dem"))
    c = override(c, "dashboard.tiles_dir", str(tmp_path / "tiles"))

    class NoDetector:
        def detect(self, frame, frame_t, tile):
            return []

    a = App(c, FrameSlot(), NoDetector(), StreamStats())
    yield a
    a.logger.close()


def test_view_only_screens_cannot_lock_or_engage_but_can_release(app_obj):
    async def go():
        lock = await app_obj.on_command("lock", {"track_id": 1}, control=False)
        engage = await app_obj.on_command("engage", {}, control=False)
        release = await app_obj.on_command("release", {}, control=False)
        return lock, engage, release

    lock, engage, release = asyncio.run(go())
    assert not lock["ok"] and "view-only" in lock["message"]
    assert not engage["ok"]
    assert release["ok"]


def test_http_endpoints(app_obj, tmp_path):
    from fastapi.testclient import TestClient

    from wildtrack.dashboard.server import TileStore, create_app

    tdir = tmp_path / "tiles" / "3" / "4"
    tdir.mkdir(parents=True)
    (tdir / "5.png").write_bytes(b"\x89PNG fake")
    tiles = TileStore(tmp_path / "tiles", None)
    client = TestClient(create_app(app_obj.hub, tiles, app_obj.export))
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/vendor/leaflet/leaflet.js").status_code == 200
    assert client.get("/tiles/3/4/5.png").content == b"\x89PNG fake"
    assert client.get("/tiles/3/4/6.png").status_code == 404
    svg = client.get("/api/qr.svg", params={"text": "geo:12.29,76.64"})
    assert svg.status_code == 200 and b"<svg" in svg.content
    gpx = client.get("/api/export.gpx")
    assert gpx.status_code == 200 and b"<gpx" in gpx.content
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["control"] is False  # TestClient is not localhost
        ws.send_json({"type": "cmd", "action": "engage"})
        reply = ws.receive_json()
        assert reply["type"] == "reply" and not reply["ok"]
