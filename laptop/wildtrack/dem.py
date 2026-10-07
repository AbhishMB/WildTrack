"""Offline terrain heights from Copernicus GLO-30 GeoTIFF tiles.

DJI altitude is relative to the takeoff point and the ground is not flat, so geo.py
needs the terrain height where each ray lands. Tiles are 1x1 degree GeoTIFFs in
geographic coordinates (tools/fetch_dem.py downloads them). Copernicus heights are
relative to the EGM2008 geoid; only differences against the home point are used,
so the datum cancels out.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_TAG_GEOKEYS = 34735
_GEOKEY_RASTER_TYPE = 1025
_RASTER_PIXEL_IS_POINT = 2


@dataclass
class DemTile:
    heights: np.ndarray  # rows from north to south
    lon0: float  # longitude of column 0's centre
    lat0: float  # latitude of row 0's centre
    dlon: float  # degrees per column (> 0)
    dlat: float  # degrees per row (> 0, rows go south)
    nodata: float | None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        rows, cols = self.heights.shape
        return (
            self.lat0 - (rows - 1) * self.dlat,
            self.lat0,
            self.lon0,
            self.lon0 + (cols - 1) * self.dlon,
        )

    def contains(self, lat: float, lon: float) -> bool:
        """Inside the sample grid, or less than one pixel outside it: neighbouring tiles
        whose samples sit on whole degrees leave a one-pixel seam between them."""
        s, n, w, e = self.bounds
        return s - self.dlat < lat < n + self.dlat and w - self.dlon < lon < e + self.dlon

    def height(self, lat: float, lon: float) -> float | None:
        """Bilinear interpolation between the four surrounding samples."""
        if not self.contains(lat, lon):
            return None
        rows, cols = self.heights.shape
        x = min(max((lon - self.lon0) / self.dlon, 0.0), cols - 1.0)
        y = min(max((self.lat0 - lat) / self.dlat, 0.0), rows - 1.0)
        c0, r0 = min(int(x), cols - 2), min(int(y), rows - 2)
        fx, fy = x - c0, y - r0
        q = self.heights[r0 : r0 + 2, c0 : c0 + 2].astype(np.float64)
        if self.nodata is not None and np.any(q == self.nodata):
            return None
        if not np.all(np.isfinite(q)):
            return None
        top = q[0, 0] * (1 - fx) + q[0, 1] * fx
        bot = q[1, 0] * (1 - fx) + q[1, 1] * fx
        return float(top * (1 - fy) + bot * fy)


def read_geotiff(path: Path) -> DemTile:
    import tifffile  # imported here so the pure modules never need it

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        heights = page.asarray()
        tags = page.tags
        sx, sy = tags[_TAG_PIXEL_SCALE].value[:2]
        tp = tags[_TAG_TIEPOINT].value
        i, j, x, y = tp[0], tp[1], tp[3], tp[4]
        pixel_is_point = False
        if _TAG_GEOKEYS in tags:
            keys = tags[_TAG_GEOKEYS].value
            for k in range(4, len(keys), 4):
                if keys[k] == _GEOKEY_RASTER_TYPE and keys[k + 1] == 0:
                    pixel_is_point = keys[k + 3] == _RASTER_PIXEL_IS_POINT
        nodata = None
        if 42113 in tags:  # GDAL_NODATA
            try:
                nodata = float(str(tags[42113].value).strip("\x00 "))
            except ValueError:
                nodata = None
    if heights.ndim != 2:
        heights = heights[..., 0]
    half = 0.0 if pixel_is_point else 0.5
    lon0 = x + (half - i) * sx
    lat0 = y - (half - j) * sy
    return DemTile(heights, lon0, lat0, float(sx), float(sy), nodata)


class Terrain:
    """Callable (lat, lon) -> height or None over all tiles in a directory."""

    def __init__(self, tiles: list[DemTile]):
        self.tiles = tiles
        self._last: DemTile | None = None

    @staticmethod
    def from_dir(dem_dir: Path) -> "Terrain | None":
        files = sorted(Path(dem_dir).glob("*.tif")) + sorted(Path(dem_dir).glob("*.tiff"))
        if not files:
            log.warning("no DEM tiles in %s: assuming flat ground (geolocation error grows on slopes)", dem_dir)
            return None
        tiles = []
        for f in files:
            try:
                tiles.append(read_geotiff(f))
                log.info("DEM tile %s bounds %s", f.name, tiles[-1].bounds)
            except Exception as exc:  # a broken tile must not stop the app
                log.error("cannot read DEM tile %s: %s", f, exc)
        return Terrain(tiles) if tiles else None

    def __call__(self, lat: float, lon: float) -> float | None:
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        if self._last is not None and self._last.contains(lat, lon):
            return self._last.height(lat, lon)
        for t in self.tiles:
            if t.contains(lat, lon):
                self._last = t
                return t.height(lat, lon)
        return None


def copernicus_tile_name(lat: float, lon: float) -> str:
    """Name of the GLO-30 tile covering a point, e.g. Copernicus_DSM_COG_10_N12_00_E076_00_DEM."""
    la, lo = math.floor(lat), math.floor(lon)
    ns = f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
    ew = f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
