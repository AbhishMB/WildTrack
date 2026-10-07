"""Download map tiles for an area into laptop/tiles/{z}/{x}/{y}.png for the offline dashboard map.

    python tools/fetch_tiles.py --lat 12.2958 --lon 76.6394 --radius-km 3 \
        --zoom 12-18 --url "https://YOUR-PROVIDER/{z}/{x}/{y}.png"

Use a tile source whose terms allow downloading for offline use: your own tile server,
a commercial provider with an offline licence, or tiles rendered in QGIS
(Processing > "Generate XYZ tiles (Directory)") from imagery you are licensed to use.
The OpenStreetMap Foundation's public tile servers forbid bulk downloading, so they are
not a default here. Alternatively pass the dashboard a single .mbtiles file
(dashboard.mbtiles in config.yaml).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def tile_xy(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 1 << z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--radius-km", type=float, default=3.0)
    p.add_argument("--zoom", default="12-18", help="e.g. 12-18")
    p.add_argument("--url", required=True, help="template with {z} {x} {y}")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "tiles"))
    p.add_argument("--delay-s", type=float, default=0.2, help="pause between requests")
    p.add_argument("--max-tiles", type=int, default=20000)
    a = p.parse_args()
    z0, z1 = (int(v) for v in a.zoom.split("-"))
    dlat = a.radius_km / 111.32
    dlon = a.radius_km / (111.32 * math.cos(math.radians(a.lat)))
    jobs = []
    for z in range(z0, z1 + 1):
        x0, y0 = tile_xy(a.lat + dlat, a.lon - dlon, z)
        x1, y1 = tile_xy(a.lat - dlat, a.lon + dlon, z)
        jobs += [(z, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    if len(jobs) > a.max_tiles:
        print(f"{len(jobs)} tiles is more than --max-tiles {a.max_tiles}; shrink the area or zoom range")
        return 1
    print(f"{len(jobs)} tiles")
    out = Path(a.out)
    for k, (z, x, y) in enumerate(jobs):
        dest = out / str(z) / str(x) / f"{y}.png"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(a.url.format(z=z, x=x, y=y), headers={"User-Agent": "wildtrack-offline-cache/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                dest.write_bytes(r.read())
        except urllib.error.URLError as e:
            print(f"  {z}/{x}/{y}: {e}")
        if k % 100 == 0:
            print(f"  {k}/{len(jobs)}")
        time.sleep(a.delay_s)
    print(f"tiles in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
