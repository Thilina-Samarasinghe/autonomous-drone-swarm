"""Terrain height map and threat zones (Phase 7).

Height
  A smooth landscape: the sum of Gaussian hills
      h(x, y) = sum_i A_i * exp(-|p - c_i|^2 / (2 s_i^2))
  evaluated analytically (no grid), so any point is exact and cheap.
  Drones fly a fixed clearance above the ground, so flying over a hill means
  climbing. Ground higher than `ceiling` is too high to fly over: those
  peaks are no-fly areas (and are also added to the world as physical
  obstacles so the lidar and ORCA respect them).

Threat zones
  Circles (e.g. radar or hazard areas) with radius R. Risk grows toward
  the centre:  risk(d) = (1 - d / R)^2  for d < R,  and the inner
  `lethal_fraction * R` core must never be entered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Terrain:
    hills: np.ndarray          # (H, 4): cx, cy, amplitude (m), sigma (m)
    threats: np.ndarray        # (T, 3): cx, cy, radius (m)
    ceiling: float             # ground higher than this cannot be flown over (m)
    lethal_fraction: float     # inner part of every threat zone that is forbidden
    _tables: dict = field(default_factory=dict, repr=False, compare=False)

    def tables(self, world_size: np.ndarray, lethal_margin: float, res: float = 0.5) -> "TerrainTables":
        """Height / risk / lethal masks sampled on a fine grid, built once and
        cached: route planning evaluates millions of points, and the terrain
        never changes."""
        key = (float(world_size[0]), float(world_size[1]), float(lethal_margin), res)
        if key not in self._tables:
            gx = int(math.ceil(world_size[0] / res)) + 1
            gy = int(math.ceil(world_size[1] / res)) + 1
            ys, xs = np.mgrid[0:gy, 0:gx] * res
            pts = np.stack([xs, ys], axis=-1)
            risk, lethal = self.risk(pts)
            _, lethal_buf = self.risk(pts, lethal_margin)
            self._tables[key] = TerrainTables(res, self.height(pts), risk, lethal, lethal_buf)
        return self._tables[key]

    def height(self, pts: np.ndarray) -> np.ndarray:
        """Ground height at points (..., 2) -> (...)."""
        if len(self.hills) == 0:
            return np.zeros(pts.shape[:-1])
        d2 = ((pts[..., None, :] - self.hills[:, :2]) ** 2).sum(-1)       # (..., H)
        return (self.hills[:, 2] * np.exp(-d2 / (2 * self.hills[:, 3] ** 2))).sum(-1)

    def risk(self, pts: np.ndarray, lethal_margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """(risk in [0, T], inside a lethal core) at points (..., 2).
        `lethal_margin` grows every lethal core by that many metres (planning buffer)."""
        if len(self.threats) == 0:
            z = np.zeros(pts.shape[:-1])
            return z, z.astype(bool)
        d = np.linalg.norm(pts[..., None, :] - self.threats[:, :2], axis=-1)   # (..., T)
        R = self.threats[:, 2]
        r = np.clip(1.0 - d / R, 0.0, None) ** 2
        lethal = (d < self.lethal_fraction * R + lethal_margin).any(-1)
        return r.sum(-1), lethal

    def no_fly_circles(self) -> np.ndarray:
        """Circles approximating the ground above `ceiling` (one per tall hill),
        radius where that hill alone reaches the ceiling. (K, 3)"""
        out = []
        for cx, cy, a, s in self.hills:
            if a > self.ceiling:
                out.append((cx, cy, s * math.sqrt(2.0 * math.log(a / self.ceiling))))
        return np.array(out, dtype=float).reshape(-1, 3)


@dataclass
class TerrainTables:
    res: float
    height: np.ndarray        # (gy, gx)
    risk: np.ndarray
    lethal: np.ndarray        # true lethal cores
    lethal_buf: np.ndarray    # lethal cores grown by the planning buffer

    def index(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gy, gx = self.height.shape
        ix = np.clip(np.rint(pts[..., 0] / self.res).astype(np.intp), 0, gx - 1)
        iy = np.clip(np.rint(pts[..., 1] / self.res).astype(np.intp), 0, gy - 1)
        return iy, ix


def generate_terrain(world_size: np.ndarray, rng: np.random.Generator, hills: int, max_height: float,
                     ceiling: float, threats: int, threat_radius: tuple[float, float],
                     lethal_fraction: float, keep_clear: np.ndarray) -> Terrain:
    """Random terrain; `keep_clear` (K, 2) points (starts / goals) stay low and safe."""
    w, h = float(world_size[0]), float(world_size[1])
    hl = []
    tries = 0
    while len(hl) < hills and tries < 5000:
        tries += 1
        c = rng.uniform((35.0, 10.0), (w - 35.0, h - 10.0))
        a = rng.uniform(0.35, 1.0) * max_height
        s = rng.uniform(7.0, 16.0)
        # the ground at every start/goal must stay well below the ceiling
        if (np.linalg.norm(keep_clear - c, axis=1) < 3.0 * s).any():
            continue
        hl.append((c[0], c[1], a, s))
    th = []
    tries = 0
    while len(th) < threats and tries < 5000:
        tries += 1
        c = rng.uniform((40.0, 12.0), (w - 40.0, h - 12.0))
        r = rng.uniform(*threat_radius)
        if (np.linalg.norm(keep_clear - c, axis=1) < r + 8.0).any():
            continue
        if any(np.hypot(*(c - q[:2])) < 0.8 * (r + q[2]) for q in th):
            continue
        th.append((c[0], c[1], r))
    return Terrain(np.array(hl, dtype=float).reshape(-1, 4), np.array(th, dtype=float).reshape(-1, 3),
                   ceiling, lethal_fraction)
