"""Synthetic camera: projects the simulated world into the image.

Gives two things: oracle detections (true boxes plus noise, dropouts and ByteTrack-like
ID changes after occlusion) for testing the control stack independently of YOLO, and
rendered frames (textured ground, trees, animal sprites) for the live dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import SimCfg
from ..geo import focal_px, project, rot_zyx
from ..models import Attitude
from .drone import SimDrone
from .world import COLOURS, Animal, World


@dataclass(frozen=True)
class SimDet:
    animal_id: int
    cls: str
    box: tuple[float, float, float, float]
    conf: float


class OracleTracker:
    """Track IDs like ByteTrack: an animal unseen for longer than the buffer comes back with a new ID."""

    def __init__(self, buffer_frames: int = 30):
        self.buffer = buffer_frames
        self.ids: dict[int, int] = {}
        self.missing: dict[int, int] = {}
        self.next_id = 1

    def assign(self, animal_ids: list[int]) -> dict[int, int]:
        seen = set(animal_ids)
        for aid in seen:
            if aid not in self.ids or self.missing.get(aid, 0) > self.buffer:
                self.ids[aid] = self.next_id
                self.next_id += 1
            self.missing[aid] = 0
        for aid in self.ids:
            if aid not in seen:
                self.missing[aid] = self.missing.get(aid, 0) + 1
        return {aid: self.ids[aid] for aid in seen}


def gimbal_attitude(d: SimDrone) -> Attitude:
    return Attitude(d.gimbal_pitch, 0.0, d.yaw)


class SimCamera:
    def __init__(self, cfg: SimCfg, hfov_deg: float):
        self.cfg = cfg
        self.w, self.h = cfg.width, cfg.height
        self.hfov = hfov_deg
        self._texture: np.ndarray | None = None

    # ------------------------------------------------------------- geometry

    def _offset(self, d: SimDrone, n: float, e: float, z: float) -> np.ndarray:
        """NED vector from the camera to a point at height z above home ground."""
        return np.array([n - d.n, e - d.e, d.h - z])

    def animal_box(self, d: SimDrone, a: Animal, world: World):
        L, W, H = a.size
        hd = math.radians(a.heading_deg)
        fn, fe = math.cos(hd), math.sin(hd)
        sn, se = -fe, fn
        g = world.ground(a.n, a.e)
        att = gimbal_attitude(d)
        pts = []
        for dz in (0.0, H):
            for fl in (-L / 2, L / 2):
                for sl in (-W / 2, W / 2):
                    p = project(self._offset(d, a.n + fl * fn + sl * sn, a.e + fl * fe + sl * se, g + dz),
                                self.w, self.h, self.hfov, att)
                    if p is None:
                        return None
                    pts.append(p)
        us = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        x1, y1, x2, y2 = min(us), min(vs), max(us), max(vs)
        cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, self.w - 1.0), min(y2, self.h - 1.0)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        if (cx2 - cx1) * (cy2 - cy1) < 0.5 * (x2 - x1) * (y2 - y1):
            return None  # mostly out of frame
        return cx1, cy1, cx2, cy2

    def detect(self, d: SimDrone, world: World, rng: np.random.Generator) -> list[SimDet]:
        out = []
        for a in world.animals:
            if not world.visible(a):
                continue
            box = self.animal_box(d, a, world)
            if box is None or rng.random() < self.cfg.det_dropout:
                continue
            j = rng.normal(0.0, self.cfg.det_noise_px, size=4)
            x1, y1, x2, y2 = (box[i] + j[i] for i in range(4))
            size = max(x2 - x1, y2 - y1)
            conf = float(np.clip(rng.normal(0.88, 0.05) - max(0.0, (25.0 - size) / 60.0), 0.2, 0.99))
            out.append(SimDet(a.id, a.cls, (x1, y1, x2, y2), conf))
        return out

    # ------------------------------------------------------------- rendering

    def texture(self, seed: int) -> np.ndarray:
        """Procedural ground: fields, scrub and dirt tracks, ground_m_per_px per pixel."""
        if self._texture is not None:
            return self._texture
        import cv2

        rng = np.random.default_rng(seed)
        size = int(2 * self.cfg.ground_extent_m / self.cfg.ground_m_per_px)
        acc = np.zeros((size, size), np.float32)
        for cells, wgt in ((6, 0.5), (24, 0.3), (96, 0.15), (384, 0.05)):
            acc += wgt * cv2.resize(rng.random((cells, cells), dtype=np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
        grass = np.array([55, 120, 75], np.float32)
        dry = np.array([95, 150, 175], np.float32)
        dark = np.array([35, 75, 45], np.float32)
        a = acc[..., None]
        img = np.where(a < 0.5, dark + (grass - dark) * (a / 0.5), grass + (dry - grass) * ((a - 0.5) / 0.5))
        img += rng.normal(0, 6, (size, size, 1)).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        for _ in range(6):
            p0 = tuple(int(x) for x in rng.integers(0, size, 2))
            p1 = tuple(int(x) for x in rng.integers(0, size, 2))
            cv2.line(img, p0, p1, (120, 165, 190), max(2, int(3 / self.cfg.ground_m_per_px)), cv2.LINE_AA)
        self._texture = img
        return img

    def _ground_homography(self, d: SimDrone) -> np.ndarray:
        """Texture pixel -> image pixel, for flat ground at height 0."""
        f = focal_px(self.w, self.hfov)
        kp = np.array([[self.w / 2.0, f, 0.0], [self.h / 2.0, 0.0, f], [1.0, 0.0, 0.0]])
        att = gimbal_attitude(d)
        r = rot_zyx(att.yaw, att.pitch, att.roll)
        m = np.array([[1.0, 0.0, -d.n], [0.0, 1.0, -d.e], [0.0, 0.0, d.h]])
        res, ext = self.cfg.ground_m_per_px, self.cfg.ground_extent_m
        t = np.array([[0.0, -res, ext], [res, 0.0, -ext], [0.0, 0.0, 1.0]])
        return kp @ r.T @ m @ t

    def _poly(self, d: SimDrone, pts_nez) -> np.ndarray | None:
        att = gimbal_attitude(d)
        out = []
        for n, e, z in pts_nez:
            p = project(self._offset(d, n, e, z), self.w, self.h, self.hfov, att)
            if p is None:
                return None
            out.append((p[0], p[1]))
        return np.array(out, np.int32)

    def render(self, d: SimDrone, world: World, seed: int) -> np.ndarray:
        import cv2

        tex = self.texture(seed)
        img = cv2.warpPerspective(tex, self._ground_homography(d), (self.w, self.h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(70, 110, 90))
        f = focal_px(self.w, self.hfov)
        horizon_v = self.h / 2.0 + f * math.tan(math.radians(d.gimbal_pitch))
        if horizon_v > -1:
            cv2.rectangle(img, (0, 0), (self.w, int(horizon_v) + 2), (215, 190, 160), -1)
        for a in world.animals:
            if not world.visible(a) and any(aid == a.id and t0 <= world.t < t1 for aid, t0, t1 in world.hide_windows):
                continue
            L, W, H = a.size
            hd = math.radians(a.heading_deg)
            fn, fe = math.cos(hd), math.sin(hd)
            sn, se = -fe, fn
            g = world.ground(a.n, a.e)
            body = [(a.n + fl * fn + sl * sn, a.e + fl * fe + sl * se, g + H)
                    for fl, sl in ((L / 2, 0.0), (L / 4, W / 2), (-L / 2, W / 2), (-L / 2, -W / 2), (L / 4, -W / 2))]
            poly = self._poly(d, body)
            if poly is not None:
                c = COLOURS.get(a.cls, (128, 128, 128))
                cv2.fillPoly(img, [poly], c, cv2.LINE_AA)
                cv2.polylines(img, [poly], True, tuple(int(x * 0.5) for x in c), 1, cv2.LINE_AA)
        for tr in world.trees:
            ring = [(tr.n + tr.r * math.cos(k * math.pi / 8), tr.e + tr.r * math.sin(k * math.pi / 8), 8.0) for k in range(16)]
            poly = self._poly(d, ring)
            if poly is not None:
                cv2.fillPoly(img, [poly], (30, 70, 30), cv2.LINE_AA)
        return img
