"""Command line: python -m wildtrack <command>

  run            fly: video + detection + phone link + dashboard
  sim            live simulator with the dashboard (or --headless for numbers only)
  bench          stage 1 bench check, propellers off
  calibrate-lag  measure how far video lags telemetry
  replay         rerun detection/geolocation/estimator on a logged sortie
  export         GPX and KML of a logged sortie
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
import time
from pathlib import Path

from . import config as config_mod


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("websockets", "uvicorn.access", "ultralytics"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_run(cfg, args) -> int:
    if cfg.lock.auto_engage:
        print("refusing to fly with lock.auto_engage: true (simulation only)", file=sys.stderr)
        return 2
    from .app import App
    from .detect import YoloDetector
    from .video_in import FrameSlot, RtspReader

    weights = cfg.path(cfg.detector.weights)
    if not weights.exists():
        if Path(cfg.detector.weights).parent == Path("."):
            weights = Path(cfg.detector.weights)  # a bare Ultralytics name, e.g. yolo26s.pt: downloaded on first use
        else:
            print(f"weights not found: {weights}. Train them (training/README.md) or set detector.weights", file=sys.stderr)
            return 2
    slot = FrameSlot()
    reader = RtspReader(cfg.video, slot)
    detector = YoloDetector(cfg.detector, weights)
    reader.start()
    app = App(cfg, slot, detector, reader.stats)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
    return 0


def cmd_sim(cfg, args) -> int:
    if args.headless:
        from .sim import closed_loop

        return closed_loop.main(cfg, args)
    from .sim.live import run_live

    print(f"simulator: open http://localhost:{cfg.dashboard.port}")
    try:
        asyncio.run(run_live(cfg, args.detector, args.auto))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_bench(cfg, args) -> int:
    """Stage 1: bridge connects, RTSP plays, telemetry at 10 Hz, altitude ~0, gimbal moves."""
    from .tello import make_link
    from .video_in import FrameSlot, RtspReader

    tels = []
    events = []

    def on_tel(tel, msg, t):
        tels.append((t, tel))

    async def go():
        link = make_link(cfg, on_tel, lambda n, r, t: events.append((n, r)), lambda u, t: None)
        task = asyncio.create_task(link.run())
        print(f"listening for {args.duration:.0f} s. Tilt the gimbal up and down during the test.")
        await asyncio.sleep(args.duration)
        task.cancel()

    slot = FrameSlot()
    reader = RtspReader(cfg.video, slot)
    reader.start()
    asyncio.run(go())
    reader.stop()

    ok = True

    def check(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    print("bench results")
    if len(tels) > 2:
        span = tels[-1][0] - tels[0][0]
        hz = (len(tels) - 1) / span if span > 0 else 0.0
        check("telemetry rate", 9.0 <= hz <= 11.0, f"{hz:.1f} Hz ({len(tels)} messages)")
        alts = [x.alt_rel_m for _, x in tels if math.isfinite(x.alt_rel_m)]
        check("altitude on the ground", alts and max(abs(a) for a in alts) < 1.0,
              f"{min(alts):.2f} .. {max(alts):.2f} m (expect ~0)" if alts else "no altitude")
        pitches = [x.gimbal.pitch for _, x in tels if math.isfinite(x.gimbal.pitch)]
        rng = (max(pitches) - min(pitches)) if pitches else 0.0
        check("gimbal attitude changes", rng > 5.0, f"pitch {min(pitches):.1f} .. {max(pitches):.1f} deg" if pitches else "none")
        last = tels[-1][1]
        print(f"  info: sats {last.gps_sats}, battery {last.battery_pct:.0f}%, mode {last.flight_mode}, home {last.home}")
    else:
        check("telemetry", False, f"no telemetry from the {cfg.link.backend} link")
    st = reader.stats
    check("video", st.frames > 0, f"{st.frames} frames, {st.width}x{st.height} at {st.fps:.1f} fps from {cfg.video.rtsp_url}")
    if st.width and st.width != 1280:
        print(f"  note: stream is {st.width} px wide; the small-object numbers in the spec assume 1280")
    if events:
        print(f"  events: {events}")
    print("next: `python -m wildtrack calibrate-lag` to measure and log the video lag")
    return 0 if ok else 1


def cmd_calibrate(cfg, args) -> int:
    from .lagcal import calibrate, write_lag

    lag, peak = asyncio.run(calibrate(cfg, args.duration))
    print(f"video lag {lag * 1000:.0f} ms (correlation peak {peak:.2f})")
    if peak < 0.5:
        print("weak peak: yaw more sharply, level the gimbal, and point at textured ground")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(cfg.path(cfg.logging.dir) / "lag_calibrations.txt", "a", encoding="utf-8") as f:
        f.write(f"{stamp} lag_s={lag:.3f} peak={peak:.2f}\n")
    if args.write:
        write_lag(Path(args.config), lag)
        print(f"wrote video.lag_s = {lag:.3f} to {args.config}")
    return 0


def cmd_replay(cfg, args) -> int:
    from .replay import main

    return main(cfg, args)


def cmd_export(cfg, args) -> int:
    import json

    from .export import to_gpx, to_kml
    from .logger import read_events
    from .models import Fix

    d = Path(args.log_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    fixes = [Fix(e["frame_t"], e["lat"], e["lon"], e["n"], e["e"], e["sigma_m"]) for e in read_events(d) if e["kind"] == "fix"]
    (d / "animal.gpx").write_text(to_gpx(fixes, meta["wall_offset_s"], d.name), encoding="utf-8")
    (d / "animal.kml").write_text(to_kml(fixes, meta["wall_offset_s"], d.name), encoding="utf-8")
    print(f"{len(fixes)} fixes -> {d / 'animal.gpx'}, {d / 'animal.kml'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wildtrack", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(config_mod.DEFAULT_PATH))
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="fly")

    s = sub.add_parser("sim", help="simulator")
    s.add_argument("--headless", action="store_true", help="virtual-time run, prints follow metrics")
    s.add_argument("--auto", action="store_true", help="auto-lock and auto-engage (live sim)")
    s.add_argument("--detector", choices=["oracle", "yolo"], default="oracle")
    s.add_argument("--duration", type=float, default=120.0)
    s.add_argument("--speed", type=float, default=None, help="animal speed, m/s")
    s.add_argument("--tau", type=float, default=None, help="total loop delay to simulate, s")
    s.add_argument("--hide", type=float, nargs=2, metavar=("FROM", "TO"), help="hide the animal between these times")
    s.add_argument("--trees", action="store_true")
    s.add_argument("--plot", help="write a top-down PNG of the run")

    b = sub.add_parser("bench", help="stage 1 bench check")
    b.add_argument("--duration", type=float, default=15.0)

    c = sub.add_parser("calibrate-lag", help="measure video lag")
    c.add_argument("--duration", type=float, default=20.0)
    c.add_argument("--write", action="store_true", help="store the result in config.yaml")

    r = sub.add_parser("replay", help="rerun a logged sortie")
    r.add_argument("log_dir")
    r.add_argument("--weights")
    r.add_argument("--logged-detections", action="store_true", help="don't rerun YOLO; use the logged boxes")
    r.add_argument("--out")

    e = sub.add_parser("export", help="GPX/KML of a logged sortie")
    e.add_argument("log_dir")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = config_mod.load(args.config)
    cfg.path(cfg.logging.dir).mkdir(parents=True, exist_ok=True)
    return {
        "run": cmd_run,
        "sim": cmd_sim,
        "bench": cmd_bench,
        "calibrate-lag": cmd_calibrate,
        "replay": cmd_replay,
        "export": cmd_export,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
