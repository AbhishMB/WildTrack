"""Loads config.yaml into frozen dataclasses.

Every threshold, gain and limit lives in config.yaml. Unknown keys, missing keys and
wrong types are errors, so a typo can never silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import logging
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LinkCfg:
    backend: str  # "dji" (phone relay) or "tello" (direct Wi-Fi UDP)
    phone_host: str
    ws_port: int
    ws_path: str
    reconnect_s: float
    telemetry_buffer_s: float
    alive_s: float
    pose_max_extrapolation_s: float

    @property
    def ws_url(self) -> str:
        return f"ws://{self.phone_host}:{self.ws_port}{self.ws_path}"


@dataclass(frozen=True)
class VideoCfg:
    rtsp_url: str
    rtsp_transport: str
    lag_s: float
    reconnect_s: float
    open_timeout_s: float


@dataclass(frozen=True)
class CameraCfg:
    hfov_deg: float


@dataclass(frozen=True)
class DetectorCfg:
    weights: str
    imgsz: int
    conf: float
    iou: float
    device: str
    half: bool
    tracker: str
    tile_search: bool
    tile_size: int
    tile_overlap: float


@dataclass(frozen=True)
class ClassesCfg:
    targets: tuple[str, ...]
    confusers: tuple[str, ...]
    test_mode: bool
    test_classes: tuple[str, ...]


@dataclass(frozen=True)
class LockCfg:
    candidate_conf: float
    candidate_frames: int
    auto_lock: bool
    auto_lock_conf: float
    auto_lock_frames: int
    auto_engage: bool
    track_lost_s: float
    track_forget_s: float
    gate_mahalanobis: float


@dataclass(frozen=True)
class GeoCfg:
    earth_radius_m: float
    dem_dir: str
    terrain_iterations: int
    sigma_gps_m: float
    sigma_att_deg: float
    sigma_alt_m: float
    min_depression_deg: float
    max_range_m: float
    nadir_pitch_deg: float


@dataclass(frozen=True)
class EstimatorCfg:
    accel_sigma: float
    init_vel_sigma: float


@dataclass(frozen=True)
class ControllerCfg:
    rate_hz: float
    standoff_m: float
    bearing_hold_m: float
    kp: float
    ki: float
    kd: float
    d_filter_s: float
    integrator_limit_mps: float
    v_max_mps: float
    a_max_mps2: float
    vz_kp: float
    vz_max_mps: float
    yaw_kp: float
    yaw_rate_max_dps: float
    gimbal_min_deg: float
    gimbal_max_deg: float
    gimbal_box_gain: float
    gimbal_rate_hz: float
    tau_s: float
    max_kp_tau: float


@dataclass(frozen=True)
class SafetyCfg:
    min_gps_sats: int
    min_battery_pct: float
    alt_min_m: float
    alt_max_m: float
    operating_radius_m: float
    no_fix_hover_s: float
    no_fix_handback_s: float
    vs_grace_s: float


@dataclass(frozen=True)
class DashboardCfg:
    host: str
    port: int
    state_hz: float
    animal_hz: float
    stale_pin_s: float
    trail_minutes: float
    tiles_dir: str
    mbtiles: str
    map_center: tuple[float, float]
    map_zoom: int
    map_max_zoom: int
    video_max_width: int
    jpeg_quality: int
    control_hosts: tuple[str, ...]


@dataclass(frozen=True)
class LoggingCfg:
    dir: str
    record_video: bool
    video_fps: float


@dataclass(frozen=True)
class TelloCfg:
    host: str
    cmd_port: int
    state_port: int
    local_cmd_port: int  # 0 = any free port
    video_port: int
    connect_retry_s: float
    state_timeout_s: float  # no state packets this long = link down
    rc_hz: float
    rc_stale_s: float  # manual stick older than this = hover
    watchdog_hover_s: float
    watchdog_handback_s: float
    rc_per_mps: float  # rc units (-100..100) per m/s; calibrate on the bench
    yaw_dps_per_unit: float  # deg/s per rc yaw unit
    max_speed_mps: float
    state_speed_mps: float  # one unit of vgx/vgy in m/s (the SDK reports dm/s)
    vg_forward_sign: int  # +1 if vgx is forward in the takeoff frame
    vg_right_sign: int  # +1 if vgy is to the right in the takeoff frame
    home: tuple[float, float]  # lat, lon anchoring the local frame; used only to report coordinates
    takeoff_heading_deg: float  # compass heading of the nose at takeoff
    camera_pitch_deg: float  # camera axis relative to the airframe, negative = down
    manual_speed_frac: float  # dashboard keys command this fraction of full stick


@dataclass(frozen=True)
class SimCfg:
    seed: int
    ws_port: int
    home: tuple[float, float]
    width: int
    height: int
    fps: float
    video_lag_s: float
    telemetry_delay_s: float
    cmd_delay_s: float
    vel_tau_s: float
    infer_s: float
    watchdog_hover_s: float
    watchdog_handback_s: float
    start_alt_m: float
    start_standoff_m: float
    start_gimbal_deg: float
    gimbal_rate_dps: float
    animal_class: str
    animal_speed_mps: float
    animal_turn_sigma_dps: float
    confusers: int
    trees: int
    tree_radius_m: float
    det_noise_px: float
    det_dropout: float
    battery_start_pct: float
    battery_drain_pct_per_min: float
    gps_sats: int
    ground_m_per_px: float
    ground_extent_m: float


@dataclass(frozen=True)
class Config:
    link: LinkCfg
    video: VideoCfg
    camera: CameraCfg
    detector: DetectorCfg
    classes: ClassesCfg
    lock: LockCfg
    geo: GeoCfg
    estimator: EstimatorCfg
    controller: ControllerCfg
    safety: SafetyCfg
    tello: TelloCfg
    dashboard: DashboardCfg
    logging: LoggingCfg
    sim: SimCfg
    base_dir: Path = field(default=Path("."), compare=False)

    def path(self, p: str) -> Path:
        """Resolve a path from the config relative to the config file."""
        q = Path(p)
        return q if q.is_absolute() else (self.base_dir / q)


def _coerce(value: Any, tp: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return _build(tp, value, where)
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported type {tp}")


def _build(cls: type, data: dict, where: str) -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.name != "base_dir"}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
    missing = names - set(data)
    if missing:
        raise ConfigError(f"{where}: missing key(s) {sorted(missing)}")
    kwargs = {n: _coerce(data[n], hints[n], f"{where}.{n}" if where else n) for n in names}
    return cls(**kwargs)


def validate(cfg: Config) -> list[str]:
    """Raise ConfigError on unsafe or inconsistent settings; return warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    c = cfg.classes
    if "person" in c.targets or "person" in c.test_classes:
        errors.append("classes: 'person' can never be a target or test class")
    if set(c.targets) & set(c.confusers):
        errors.append(f"classes: in both targets and confusers: {sorted(set(c.targets) & set(c.confusers))}")
    if cfg.link.backend not in ("dji", "tello"):
        errors.append("link.backend must be 'dji' or 'tello'")
    s = cfg.safety
    if not 0 < s.alt_min_m < s.alt_max_m:
        errors.append("safety: need 0 < alt_min_m < alt_max_m")
    if s.no_fix_hover_s >= s.no_fix_handback_s:
        errors.append("safety: no_fix_hover_s must be shorter than no_fix_handback_s")
    k = cfg.controller
    if not 5.0 <= k.rate_hz <= 25.0:
        errors.append("controller.rate_hz must be within DJI's 5-25 Hz")
    if k.gimbal_min_deg >= k.gimbal_max_deg:
        errors.append("controller: gimbal_min_deg must be below gimbal_max_deg")
    if k.kd < 0.0 or k.d_filter_s < 0.0:
        errors.append("controller: kd and d_filter_s must be >= 0")
    if k.kp * k.tau_s > k.max_kp_tau:
        warnings.append(
            f"controller: kp*tau = {k.kp * k.tau_s:.2f} rad exceeds {k.max_kp_tau} rad; "
            f"phase margin ~{90 - (k.kp * k.tau_s) * 57.3:.0f} deg. Lower kp to <= {k.max_kp_tau / k.tau_s:.2f}"
        )
    lk = cfg.lock
    if lk.auto_lock_conf < lk.candidate_conf:
        errors.append("lock: auto_lock_conf must be >= candidate_conf")
    if c.test_mode and c.test_classes:
        warnings.append(f"TEST MODE: lockable stand-in classes {list(c.test_classes)}")
    if errors:
        raise ConfigError("; ".join(errors))
    return warnings


def from_dict(data: dict, base_dir: Path = Path(".")) -> Config:
    cfg = _build(Config, data, "")
    cfg = dataclasses.replace(cfg, base_dir=base_dir)
    for w in validate(cfg):
        log.warning(w)
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_raw(p: Path) -> dict:
    """Read a YAML file; a top-level `_base: other.yaml` overlays this file on that one."""
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base", None)
    if base:
        data = _deep_merge(_load_raw((p.parent / base).resolve()), data)
    return data


def load(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_PATH
    return from_dict(_load_raw(p), base_dir=p.resolve().parent)


def override(cfg: Config, dotted: str, value: Any) -> Config:
    """Return a copy with one value replaced, e.g. override(cfg, "lock.auto_engage", True)."""
    section, key = dotted.split(".", 1)
    sub = getattr(cfg, section)
    hints = typing.get_type_hints(type(sub))
    if key not in hints:
        raise ConfigError(f"unknown key {dotted}")
    new_sub = dataclasses.replace(sub, **{key: _coerce(value, hints[key], dotted)})
    return dataclasses.replace(cfg, **{section: new_sub})
