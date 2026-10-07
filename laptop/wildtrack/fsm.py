"""Mode state machine: MANUAL -> LOCKED -> AUTO <-> LOST (docs/SPEC.md, "Mode state machine").

TABLE below is the spec's transition table, row for row. Guards read a `World`
snapshot; actions are returned to the caller (core.py), which performs them. Pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .config import SafetyCfg


class Mode(str, Enum):
    MANUAL = "MANUAL"
    LOCKED = "LOCKED"
    AUTO = "AUTO"
    LOST = "LOST"


class Ev(str, Enum):
    LOCK = "lock"  # operator clicked a box or pressed L
    AUTO_LOCK = "auto_lock"  # candidate held long enough with auto_lock on
    RELEASE = "release"  # operator: RELEASE button or Esc
    TRACK_LOST = "track_lost"  # Locked: target not seen for lock.track_lost_s
    ENGAGE = "engage"  # operator: ENGAGE button or E
    NO_FIX = "no_fix"  # Auto: no fix for safety.no_fix_hover_s
    REACQUIRED = "reacquired"  # Lost: fix of the same class inside the Kalman gate
    LOST_TIMEOUT = "lost_timeout"  # Lost: no fix for safety.no_fix_handback_s
    VS_EXIT = "vs_exit"  # DJI ended Virtual Stick (Pause, mode switch, RC loss, RTH, low battery, NFZ)
    OUT_OF_RADIUS = "out_of_radius"
    LOW_BATTERY = "low_battery"
    LAPTOP_SILENT = "laptop_silent"  # the phone's 3 s watchdog fired, or our own loop stalled
    LINK_LOST = "link_lost"  # no telemetry: the phone's watchdog will hover then hand back


class Act(str, Enum):
    FIX_TARGET = "fix_target"  # fix the track ID; start the Kalman filter and GPS publishing
    CLEAR_TARGET = "clear_target"
    VS_ENABLE = "vs_enable"  # Virtual Stick on, advanced mode on
    START_COMMANDS = "start_commands"  # 10 Hz velocity commands
    HOVER = "hover"  # command zero velocity
    RESET_INTEGRATOR = "reset_integrator"
    VS_DISABLE = "vs_disable"
    STOP_COMMANDS = "stop_commands"
    ALERT_PILOT = "alert_pilot"


@dataclass(frozen=True)
class World:
    """What the guards need to know, sampled when the event is handled."""

    target_lockable: bool = False
    auto_lock_enabled: bool = False
    gps_sats: int = 0
    battery_pct: float = 0.0
    agl_m: float | None = None
    dist_home_m: float | None = None
    link_alive: bool = False
    has_fix: bool = False  # the locked animal has a GPS estimate


Guard = Callable[[World, SafetyCfg], list[str]]  # returns refusal reasons; empty = pass


def guard_lockable(w: World, s: SafetyCfg) -> list[str]:
    return [] if w.target_lockable else ["class is not a lockable target species"]


def guard_auto_lock(w: World, s: SafetyCfg) -> list[str]:
    return guard_lockable(w, s) + ([] if w.auto_lock_enabled else ["auto_lock is off"])


def guard_engage(w: World, s: SafetyCfg) -> list[str]:
    r = []
    if w.gps_sats < s.min_gps_sats:
        r.append(f"GPS satellites {w.gps_sats} < {s.min_gps_sats}")
    if w.battery_pct < s.min_battery_pct:
        r.append(f"battery {w.battery_pct:.0f}% < {s.min_battery_pct:.0f}%")
    if w.agl_m is None:
        r.append("height above ground unknown")
    elif not s.alt_min_m <= w.agl_m <= s.alt_max_m:
        r.append(f"height {w.agl_m:.0f} m outside {s.alt_min_m:.0f}-{s.alt_max_m:.0f} m")
    if w.dist_home_m is None:
        r.append("position or home point unknown")
    elif w.dist_home_m > s.operating_radius_m:
        r.append(f"{w.dist_home_m:.0f} m from home, outside the {s.operating_radius_m:.0f} m radius")
    if not w.link_alive:
        r.append("phone link down")
    if not w.has_fix:
        r.append("no GPS fix of the animal yet")
    return r


@dataclass(frozen=True)
class Row:
    src: frozenset[Mode]
    events: frozenset[Ev]
    guard: Guard | None
    dst: Mode
    actions: tuple[Act, ...]


def _row(src, events, guard, dst, actions) -> Row:
    return Row(frozenset(src), frozenset(events), guard, dst, tuple(actions))


M, L, A, X = Mode.MANUAL, Mode.LOCKED, Mode.AUTO, Mode.LOST
_HANDBACK = (Act.VS_DISABLE, Act.STOP_COMMANDS, Act.CLEAR_TARGET)

TABLE: tuple[Row, ...] = (
    _row({M}, {Ev.LOCK}, guard_lockable, L, [Act.FIX_TARGET]),
    _row({M}, {Ev.AUTO_LOCK}, guard_auto_lock, L, [Act.FIX_TARGET]),
    _row({L}, {Ev.RELEASE, Ev.TRACK_LOST}, None, M, [Act.CLEAR_TARGET]),
    _row({L}, {Ev.ENGAGE}, guard_engage, A, [Act.VS_ENABLE, Act.RESET_INTEGRATOR, Act.START_COMMANDS]),
    _row({A}, {Ev.NO_FIX}, None, X, [Act.HOVER]),
    _row({X}, {Ev.REACQUIRED}, None, A, [Act.RESET_INTEGRATOR, Act.START_COMMANDS]),
    _row({X}, {Ev.LOST_TIMEOUT}, None, M, [*_HANDBACK, Act.ALERT_PILOT]),
    # DJI already ended Virtual Stick: stop sending and never re-enable on our own
    _row({A, X}, {Ev.VS_EXIT}, None, M, [Act.STOP_COMMANDS, Act.CLEAR_TARGET, Act.ALERT_PILOT]),
    _row({A, X}, {Ev.RELEASE}, None, M, _HANDBACK),
    _row(
        {A, X},
        {Ev.OUT_OF_RADIUS, Ev.LOW_BATTERY, Ev.LAPTOP_SILENT, Ev.LINK_LOST},
        None,
        M,
        [*_HANDBACK, Act.ALERT_PILOT],
    ),
)


@dataclass(frozen=True)
class Outcome:
    event: Ev
    src: Mode
    dst: Mode
    accepted: bool
    actions: tuple[Act, ...] = ()
    refusals: tuple[str, ...] = ()


class ModeMachine:
    def __init__(self, safety: SafetyCfg, table: tuple[Row, ...] = TABLE):
        self.safety = safety
        self.table = table
        self.mode = Mode.MANUAL

    def handle(self, ev: Ev, world: World = World()) -> Outcome:
        for row in self.table:
            if self.mode in row.src and ev in row.events:
                refusals = row.guard(world, self.safety) if row.guard else []
                if refusals:
                    return Outcome(ev, self.mode, self.mode, False, (), tuple(refusals))
                src, self.mode = self.mode, row.dst
                return Outcome(ev, src, row.dst, True, row.actions)
        return Outcome(ev, self.mode, self.mode, False, (), (f"{ev.value} does nothing in {self.mode.value}",))
