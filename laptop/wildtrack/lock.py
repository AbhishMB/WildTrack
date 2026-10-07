"""Candidate / lock / re-acquire rules (docs/SPEC.md, "Lock rules").

Which classes may be locked is decided here, not in the UI: confusers can never be
locked, and a person can never be locked in any mode, test mode included. Pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .config import ClassesCfg, LockCfg
from .models import Detection

# Safety invariant, deliberately not configurable: the drone never chases a person.
NEVER_LOCK = frozenset({"person"})


def lockable_classes(c: ClassesCfg) -> frozenset[str]:
    allowed = set(c.targets) - set(c.confusers)
    if c.test_mode:
        allowed |= set(c.test_classes)
    return frozenset(allowed - NEVER_LOCK)


@dataclass
class TrackStats:
    cls: str
    streak: int = 0  # consecutive frames at >= candidate_conf
    auto_streak: int = 0  # consecutive frames at >= auto_lock_conf
    last_seen: float = 0.0
    last_conf: float = 0.0


class LockManager:
    def __init__(self, cfg: LockCfg, classes: ClassesCfg):
        self.cfg = cfg
        self.lockable = lockable_classes(classes)
        self.stats: dict[int, TrackStats] = {}
        self.locked_id: int | None = None
        self.locked_cls: str | None = None

    def is_lockable(self, cls: str) -> bool:
        return cls in self.lockable

    def observe(self, dets: Iterable[Detection], frame_t: float) -> None:
        """Update per-track streaks with one frame's detections."""
        seen = set()
        for d in dets:
            seen.add(d.track_id)
            s = self.stats.get(d.track_id)
            if s is None or s.cls != d.cls:
                s = self.stats[d.track_id] = TrackStats(d.cls)
            s.streak = s.streak + 1 if d.conf >= self.cfg.candidate_conf else 0
            s.auto_streak = s.auto_streak + 1 if d.conf >= self.cfg.auto_lock_conf else 0
            s.last_seen, s.last_conf = frame_t, d.conf
        for tid, s in list(self.stats.items()):
            if tid not in seen:
                s.streak = s.auto_streak = 0  # "consecutive" frames
                if frame_t - s.last_seen > self.cfg.track_forget_s:
                    del self.stats[tid]

    def is_candidate(self, d: Detection) -> bool:
        s = self.stats.get(d.track_id)
        return (
            self.is_lockable(d.cls)
            and s is not None
            and s.cls == d.cls
            and s.streak >= self.cfg.candidate_frames
        )

    def candidates(self, dets: Iterable[Detection]) -> list[Detection]:
        return [d for d in dets if self.is_candidate(d)]

    def best_candidate(self, dets: Iterable[Detection]) -> Detection | None:
        """The L key: the highest-confidence candidate."""
        c = self.candidates(dets)
        return max(c, key=lambda d: d.conf) if c else None

    def auto_lock_ready(self, dets: Iterable[Detection]) -> Detection | None:
        ready = [
            d
            for d in self.candidates(dets)
            if self.stats[d.track_id].auto_streak >= self.cfg.auto_lock_frames
        ]
        return max(ready, key=lambda d: d.conf) if ready else None

    def lock(self, d: Detection) -> None:
        if not self.is_lockable(d.cls):  # second line of defence behind the FSM guard
            raise ValueError(f"refusing to lock class {d.cls!r}")
        self.locked_id, self.locked_cls = d.track_id, d.cls

    def release(self) -> None:
        self.locked_id = self.locked_cls = None

    @property
    def locked(self) -> bool:
        return self.locked_id is not None

    def find_target(self, dets: Iterable[Detection]) -> Detection | None:
        """The locked animal in this frame, by track ID."""
        for d in dets:
            if d.track_id == self.locked_id:
                return d
        return None

    def reacquire(
        self, dets: Iterable[Detection], gate_distance: Callable[[Detection], float | None]
    ) -> Detection | None:
        """ByteTrack dropped the ID: take the nearest same-class detection inside the Kalman gate.

        gate_distance returns the Mahalanobis distance of a detection's geolocated
        position from the prediction, or None if it cannot be geolocated.
        """
        best, best_d = None, self.cfg.gate_mahalanobis
        for d in dets:
            if d.cls != self.locked_cls or d.track_id == self.locked_id:
                continue
            dist = gate_distance(d)
            if dist is not None and dist < best_d:
                best, best_d = d, dist
        if best is not None:
            self.locked_id = best.track_id
        return best
