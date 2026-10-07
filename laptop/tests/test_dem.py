import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from wildtrack.dem import Terrain, copernicus_tile_name, read_geotiff  # noqa: E402


def write_tile(path, pixel_is_point: bool):
    rows = np.arange(11, dtype=np.float32)[:, None]
    cols = np.arange(11, dtype=np.float32)[None, :]
    data = (rows * 10 + cols).astype(np.float32)  # height = 10 * row + col
    tags = [
        (33550, "d", 3, (0.1, 0.1, 0.0), True),
        (33922, "d", 6, (0.0, 0.0, 0.0, 76.0, 13.0, 0.0), True),
    ]
    if pixel_is_point:
        tags.append((34735, "H", 8, (1, 1, 0, 1, 1025, 0, 1, 2), True))
    tifffile.imwrite(path, data, extratags=tags, compression="zlib")


def test_pixel_is_area(tmp_path):
    p = tmp_path / "a.tif"
    write_tile(p, pixel_is_point=False)
    t = read_geotiff(p)
    # pixel centres are half a pixel in from the tie point
    assert t.lon0 == pytest.approx(76.05) and t.lat0 == pytest.approx(12.95)
    assert t.height(12.95, 76.05) == pytest.approx(0.0)
    assert t.height(12.85, 76.15) == pytest.approx(11.0)  # row 1, col 1
    assert t.height(12.90, 76.10) == pytest.approx(5.5)  # bilinear midpoint
    assert t.height(13.5, 76.1) is None
    assert t.height(12.96, 76.05) == pytest.approx(0.0)  # within a pixel of the edge: clamped, bridges tile seams


def test_pixel_is_point(tmp_path):
    p = tmp_path / "b.tif"
    write_tile(p, pixel_is_point=True)
    t = read_geotiff(p)
    assert t.lon0 == pytest.approx(76.0) and t.lat0 == pytest.approx(13.0)
    assert t.height(12.9, 76.1) == pytest.approx(11.0)


def test_terrain_dir(tmp_path):
    write_tile(tmp_path / "a.tif", pixel_is_point=False)
    terrain = Terrain.from_dir(tmp_path)
    assert terrain(12.85, 76.15) == pytest.approx(11.0)
    assert terrain(20.0, 70.0) is None
    assert Terrain.from_dir(tmp_path / "missing") is None


def test_tile_names():
    assert copernicus_tile_name(12.29, 76.64) == "Copernicus_DSM_COG_10_N12_00_E076_00_DEM"
    assert copernicus_tile_name(-0.5, -70.2) == "Copernicus_DSM_COG_10_S01_00_W071_00_DEM"
