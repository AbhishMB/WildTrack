"""Download the Copernicus GLO-30 elevation tiles covering an area, for offline use.

    python tools/fetch_dem.py --lat 12.2958 --lon 76.6394 --radius-km 5

Tiles come from the Copernicus DEM on AWS Open Data (no account needed) and are saved
to laptop/dem/, where the app looks for them (geo.dem_dir). Run this before going to
the field. Each 1x1 degree tile is about 20-50 MB.
"""

from __future__ import annotations

import argparse
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wildtrack.dem import copernicus_tile_name  # noqa: E402

URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def tiles_for(lat: float, lon: float, radius_km: float) -> list[str]:
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(0.01, math.cos(math.radians(lat))))
    names = []
    for la in range(math.floor(lat - dlat), math.floor(lat + dlat) + 1):
        for lo in range(math.floor(lon - dlon), math.floor(lon + dlon) + 1):
            names.append(copernicus_tile_name(la + 0.5, lo + 0.5))
    return names


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--radius-km", type=float, default=5.0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "dem"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in tiles_for(a.lat, a.lon, a.radius_km):
        dest = out / f"{name}.tif"
        if dest.exists():
            print(f"have {dest.name}")
            continue
        url = URL.format(name=name)
        print(f"fetching {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest.with_suffix(".part"), "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            dest.with_suffix(".part").replace(dest)
        except urllib.error.HTTPError as e:
            print(f"  {e.code}: no tile (open sea has none)")
    print(f"DEM tiles in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
