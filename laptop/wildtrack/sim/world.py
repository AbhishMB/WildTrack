"""Simulated ground: animals on random walks, and trees they can hide under."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config import SimCfg

# length, width, height in metres (rough adult sizes)
SIZES = {
    "elephant": (5.5, 2.5, 3.0),
    "leopard": (1.6, 0.5, 0.7),
    "tiger": (2.6, 0.8, 1.0),
    "wild_boar": (1.4, 0.6, 0.9),
    "gaur": (3.0, 1.2, 1.8),
    "sloth_bear": (1.6, 0.8, 0.9),
    "deer": (1.6, 0.5, 1.1),
    "cattle": (2.4, 0.8, 1.4),
    "goat": (1.1, 0.4, 0.7),
    "dog": (0.9, 0.3, 0.6),
    "person": (0.5, 0.4, 1.7),
    "tarp": (3.0, 2.0, 0.05),
}

# BGR colours the synthetic camera paints each class with
COLOURS = {
    "elephant": (110, 110, 120),
    "leopard": (40, 150, 210),
    "tiger": (20, 110, 230),
    "wild_boar": (40, 50, 70),
    "gaur": (30, 30, 40),
    "sloth_bear": (15, 15, 15),
    "deer": (60, 110, 160),
    "cattle": (230, 230, 230),
    "goat": (200, 200, 190),
    "dog": (90, 140, 180),
    "person": (200, 60, 60),
    "tarp": (200, 120, 20),
}

GroundFn = Callable[[float, float], float]  # (n, e) -> height above home ground, metres


@dataclass
class Animal:
    id: int
    cls: str
    n: float
    e: float
    heading_deg: float
    speed: float
    turn_sigma_dps: float
    hidden_until: float = -1.0  # forced occlusion (scenario), sim time

    @property
    def size(self) -> tuple[float, float, float]:
        return SIZES.get(self.cls, (2.0, 1.0, 1.0))

    @property
    def vn(self) -> float:
        return self.speed * math.cos(math.radians(self.heading_deg))

    @property
    def ve(self) -> float:
        return self.speed * math.sin(math.radians(self.heading_deg))

    def step(self, dt: float, rng: np.random.Generator) -> None:
        self.heading_deg = (self.heading_deg + rng.normal(0.0, self.turn_sigma_dps * math.sqrt(dt))) % 360.0
        self.n += self.vn * dt
        self.e += self.ve * dt


@dataclass
class Tree:
    n: float
    e: float
    r: float


@dataclass
class World:
    animals: list[Animal]
    trees: list[Tree]
    ground: GroundFn = field(default=lambda n, e: 0.0)
    t: float = 0.0
    hide_windows: list[tuple[int, float, float]] = field(default_factory=list)  # (animal id, from, to)

    @property
    def target(self) -> Animal:
        return self.animals[0]

    def step(self, dt: float, rng: np.random.Generator) -> None:
        self.t += dt
        for a in self.animals:
            a.step(dt, rng)

    def visible(self, a: Animal) -> bool:
        for aid, t0, t1 in self.hide_windows:
            if aid == a.id and t0 <= self.t < t1:
                return False
        return not any(math.hypot(a.n - tr.n, a.e - tr.e) < tr.r for tr in self.trees)


def make_world(
    cfg: SimCfg,
    rng: np.random.Generator,
    speed: float | None = None,
    trees: bool = True,
    ground: GroundFn | None = None,
) -> World:
    """The target walks away from the home area; a few confusers wander nearby."""
    target = Animal(
        id=1,
        cls=cfg.animal_class,
        n=60.0,
        e=40.0,
        heading_deg=float(rng.uniform(0, 360)),
        speed=cfg.animal_speed_mps if speed is None else speed,
        turn_sigma_dps=cfg.animal_turn_sigma_dps,
    )
    animals = [target]
    for i in range(cfg.confusers):
        ang = rng.uniform(0, 2 * math.pi)
        animals.append(
            Animal(
                id=100 + i,
                cls="cattle",
                n=target.n + 50 * math.cos(ang),
                e=target.e + 50 * math.sin(ang),
                heading_deg=float(rng.uniform(0, 360)),
                speed=0.4,
                turn_sigma_dps=20.0,
            )
        )
    tree_list: list[Tree] = []
    if trees:
        ext = cfg.ground_extent_m
        while len(tree_list) < cfg.trees:
            n, e = rng.uniform(-ext, ext, size=2)
            if math.hypot(n - target.n, e - target.e) > 3 * cfg.tree_radius_m:
                tree_list.append(Tree(float(n), float(e), cfg.tree_radius_m * float(rng.uniform(0.6, 1.3))))
    return World(animals, tree_list, ground or (lambda n, e: 0.0))
