"""Victims and their radio signal.

Each victim carries a phone that transmits with its own ID. A drone's
receiver measures the strength of every ID it can hear:

    strength = 1 / (1 + (d / 10 m)^2)  +  gaussian noise

so it is ~1 right next to the victim, ~0.2 at 20 m and ~0.05 at 45 m, where
the noise (std 0.01 by default) makes readings unreliable. Beyond
`signal_range` nothing is heard. Obstacles do not block the signal (a
simplification). A victim is *confirmed* only when a drone's camera sees
it, i.e. the drone is within `detect_radius`.
"""

from __future__ import annotations

import numpy as np

REFERENCE_DISTANCE = 10.0
HEAR_THRESHOLD = 0.03       # readings below this are treated as "not heard"


class VictimField:
    def __init__(self, positions: np.ndarray, signal_range: float, noise: float,
                 rng: np.random.Generator):
        self.pos = np.asarray(positions, dtype=float).reshape(-1, 2)
        self.signal_range = signal_range
        self.noise = noise
        self.rng = rng

    @property
    def count(self) -> int:
        return len(self.pos)

    def distances(self, drone_pos: np.ndarray) -> np.ndarray:
        """(N, V) true distances (used for camera confirmation and metrics only)."""
        return np.linalg.norm(drone_pos[:, None] - self.pos[None], axis=2)

    def true_strength(self, d: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + (d / REFERENCE_DISTANCE) ** 2)

    def read(self, drone_pos: np.ndarray) -> np.ndarray:
        """(N, V) noisy signal strength per victim ID; 0 where not heard."""
        d = self.distances(drone_pos)
        s = self.true_strength(d) + self.rng.normal(0.0, self.noise, d.shape)
        s[(d > self.signal_range) | (s < HEAR_THRESHOLD)] = 0.0
        return s
