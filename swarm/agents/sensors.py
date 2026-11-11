"""Onboard range sensing (a simple 2D lidar).

Drones never read the obstacle list directly: everything they know about
obstacles comes from these ray distances. `closest_points` turns a scan into
a few "nearest obstacle point" detections (local minima of the scan), which
is what a real drone gets by clustering its lidar returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.sim.world import World


@dataclass
class Detections:
    """One entry per detected obstacle point (local minimum of a scan)."""
    i: np.ndarray        # (K,) drone index
    dir: np.ndarray      # (K, 2) unit vector from drone toward the point
    dist: np.ndarray     # (K,) distance from drone centre to the point

    def within(self, d: float) -> "Detections":
        m = self.dist < d
        return Detections(self.i[m], self.dir[m], self.dist[m])


@dataclass
class Scan:
    dirs: np.ndarray       # (R, 2) unit beam directions
    dist: np.ndarray       # (N, R) distance per beam (== max_range on miss)
    max_range: float

    @property
    def hits(self) -> np.ndarray:
        return self.dist < self.max_range - 1e-9

    def closest_points(self) -> Detections:
        d = self.dist
        hit = self.hits
        left = np.roll(d, 1, axis=1)
        right = np.roll(d, -1, axis=1)
        is_min = hit & (d <= left) & (d <= right)
        i, r = np.nonzero(is_min)
        return Detections(i, self.dirs[r], d[i, r])


class RangeSensor:
    def __init__(self, num_rays: int, max_range: float):
        ang = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        self.dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.max_range = max_range

    def scan(self, pos: np.ndarray, world: World) -> Scan:
        if world.obstacles.count == 0 or len(pos) == 0:
            dist = np.full((len(pos), len(self.dirs)), self.max_range)
        else:
            dist = world.obstacles.raycast(pos, self.dirs, self.max_range)
        return Scan(self.dirs, dist, self.max_range)
