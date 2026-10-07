"""GPX and KML of the animal's track, per sortie. Pure: returns strings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
from xml.sax.saxutils import escape

from .models import Fix


def _iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def to_gpx(fixes: Sequence[Fix], wall_offset_s: float, name: str) -> str:
    """wall_offset_s converts laptop monotonic time to Unix time (wall = mono + offset)."""
    pts = "\n".join(
        f'      <trkpt lat="{f.lat:.7f}" lon="{f.lon:.7f}"><time>{_iso(f.t + wall_offset_s)}</time>'
        f"<hdop>{f.sigma_m:.1f}</hdop><desc>1-sigma {f.sigma_m:.1f} m</desc></trkpt>"
        for f in fixes
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="wildtrack" xmlns="http://www.topografix.com/GPX/1/1">\n'
        f"  <metadata><name>{escape(name)}</name></metadata>\n"
        f"  <trk><name>{escape(name)}</name>\n    <trkseg>\n{pts}\n    </trkseg>\n  </trk>\n</gpx>\n"
    )


def to_kml(fixes: Sequence[Fix], wall_offset_s: float, name: str) -> str:
    coords = " ".join(f"{f.lon:.7f},{f.lat:.7f},0" for f in fixes)
    last = ""
    if fixes:
        f = fixes[-1]
        last = (
            "    <Placemark>\n"
            f"      <name>Last sighting {_iso(f.t + wall_offset_s)}</name>\n"
            f"      <description>1-sigma radius {f.sigma_m:.1f} m</description>\n"
            f"      <Point><coordinates>{f.lon:.7f},{f.lat:.7f},0</coordinates></Point>\n"
            "    </Placemark>\n"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n  <Document>\n'
        f"    <name>{escape(name)}</name>\n"
        '    <Style id="trk"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>\n'
        "    <Placemark>\n      <name>Animal track</name>\n      <styleUrl>#trk</styleUrl>\n"
        f"      <LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString>\n"
        "    </Placemark>\n"
        f"{last}  </Document>\n</kml>\n"
    )
