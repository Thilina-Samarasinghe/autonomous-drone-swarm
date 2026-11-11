"""Swarm state stored as structure-of-arrays for fast vectorised updates.

Each row is one drone. Controllers only ever *read* neighbour data inside a
drone's perception radius, so decisions stay decentralised even though the
maths is batched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.sim.world import World


@dataclass
class SwarmState:
    pos: np.ndarray   # (N, 2) metres
    vel: np.ndarray   # (N, 2) m/s
    ids: np.ndarray   # (N,) stable drone ids

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    def copy(self) -> "SwarmState":
        return SwarmState(self.pos.copy(), self.vel.copy(), self.ids.copy())

    @classmethod
    def spawn(
        cls,
        n: int,
        world: World,
        rng: np.random.Generator,
        spacing: float,
        speed: float,
        region: tuple[float, float, float, float] | None = None,
        obstacle_clearance: float = 3.0,
    ) -> "SwarmState":
        """Random non-overlapping positions inside `region` (x0, y0, x1, y1),
        at least `obstacle_clearance` metres from any obstacle.

        Defaults to the central half of the arena. Initial headings are random.
        """
        if region is None:
            region = (world.width * 0.25, world.height * 0.25,
                      world.width * 0.75, world.height * 0.75)

        def free(p: np.ndarray) -> bool:
            if world.obstacles.count == 0:
                return True
            sd, _ = world.obstacles.signed_distance(p[None])
            return bool(sd[0] >= obstacle_clearance)

        pos = _sample_spaced(n, region, spacing, rng, free)
        heading = rng.uniform(0, 2 * np.pi, n)
        vel = np.stack([np.cos(heading), np.sin(heading)], axis=1) * speed
        return cls(pos, vel, np.arange(n))

    def add(self, pos: np.ndarray, vel: np.ndarray) -> None:
        start = int(self.ids.max()) + 1 if self.n else 0
        self.pos = np.vstack([self.pos, pos])
        self.vel = np.vstack([self.vel, vel])
        self.ids = np.concatenate([self.ids, np.arange(start, start + len(pos))])

    def remove_last(self, k: int = 1) -> None:
        keep = max(self.n - k, 0)
        self.pos, self.vel, self.ids = self.pos[:keep], self.vel[:keep], self.ids[:keep]


def _sample_spaced(n, region, spacing, rng, valid=None, max_tries: int = 200_000) -> np.ndarray:
    x0, y0, x1, y1 = region
    pts: list[np.ndarray] = []
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"Could not place {n} drones with spacing {spacing} in region {region}")
        p = rng.uniform((x0, y0), (x1, y1))
        if all(np.hypot(*(p - q)) >= spacing for q in pts) and (valid is None or valid(p)):
            pts.append(p)
    return np.array(pts, dtype=float).reshape(n, 2)
