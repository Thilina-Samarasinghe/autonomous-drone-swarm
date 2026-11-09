"""The 2D environment. Phase 1: rectangular arena. Obstacles arrive in Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.config import WorldConfig


@dataclass
class World:
    width: float
    height: float
    boundary_margin: float

    @classmethod
    def from_config(cls, cfg: WorldConfig) -> "World":
        return cls(cfg.width, cfg.height, cfg.boundary_margin)

    @property
    def size(self) -> np.ndarray:
        return np.array([self.width, self.height])

    def boundary_push(self, pos: np.ndarray) -> np.ndarray:
        """Inward push in [0, 1] per axis, growing linearly inside the margin.

        Returns (N, 2); zero for drones well inside the arena.
        """
        m = self.boundary_margin
        low = np.clip((m - pos) / m, 0.0, 1.0)
        high = np.clip((pos - (self.size - m)) / m, 0.0, 1.0)
        return low - high

    def enforce_bounds(self, pos: np.ndarray, vel: np.ndarray, radius: float) -> None:
        """Hard walls (in place): clamp position and cancel outward velocity."""
        lo = radius
        hi = self.size - radius
        below = pos < lo
        above = pos > hi
        pos[:] = np.clip(pos, lo, hi)
        vel[below & (vel < 0)] = 0.0
        vel[above & (vel > 0)] = 0.0
