"""Per-drone observations for learned policies (identical in training and deployment).

Every drone sees only local, egocentric information (55 numbers):

  [0:2]    own velocity / max_speed
  [2:4]    direction to its target, scaled by 1/TARGET_SCALE and clipped to length 1
  [4]      tanh(distance to target / TARGET_SCALE)
  [5:35]   6 nearest neighbours within perception range, nearest first:
           relative position / perception_radius, relative velocity / max_speed,
           present flag (zeros when fewer neighbours)
  [35:47]  lidar: 12 sectors, nearest hit / sensor range (1 = nothing seen),
           where the arena edge (a geofence the drone knows from GPS) also
           counts as a hit. Without it the policy is penalised for walls it
           cannot perceive, and learning goes backwards.
  [47:55]  own map: fraction of still-unknown cells in 8 sectors within
           SECTOR_RADIUS (zeros when the drone has no map)

The "target" depends on the task: a goal or planner waypoint (navigate), a
formation slot (formation), nothing (flock, search).

Actions are DESIRED VELOCITIES in [-1, 1]^2 x max_speed; `action_to_accel`
turns them into an acceleration with a fixed velocity-tracking rule. (An
earlier version used raw accelerations: then every task is also a momentum
control problem, and nothing learned in the CPU budget.)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

K_NEIGHBORS = 6
LIDAR_SECTORS = 12
MAP_SECTORS = 8
SECTOR_RADIUS = 20.0
TARGET_SCALE = 20.0
OBS_DIM = 5 + 5 * K_NEIGHBORS + LIDAR_SECTORS + MAP_SECTORS
ACTION_TAU = 0.5          # s: how fast a drone tries to reach the commanded velocity


def action_to_accel(action: np.ndarray, vel: np.ndarray, max_speed: float, max_accel: float) -> np.ndarray:
    """Desired-velocity command (N, 2) in [-1, 1] -> acceleration (N, 2), clamped."""
    desired = np.clip(action, -1.0, 1.0) * max_speed
    acc = (desired - vel) / ACTION_TAU
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    return acc * np.minimum(1.0, max_accel / np.maximum(n, 1e-9))


class Observer:
    def __init__(self, max_speed: float, perception: float, maps=None,
                 world_size: np.ndarray | None = None):
        self.max_speed = max_speed
        self.perception = perception
        self.world_size = None if world_size is None else np.asarray(world_size, dtype=float)
        ang = (np.arange(LIDAR_SECTORS) + 0.5) * 2 * math.pi / LIDAR_SECTORS
        self._sector_dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        self._map_offsets = None
        if maps is not None:
            self._build_map_offsets(maps)

    def _build_map_offsets(self, maps) -> None:
        r = int(math.ceil(SECTOR_RADIUS / maps.cell))
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        dist = np.hypot(dx, dy) * maps.cell
        keep = (dist <= SECTOR_RADIUS) & (dist > 0)
        dx, dy = dx[keep], dy[keep]
        sector = ((np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * MAP_SECTORS).astype(int) % MAP_SECTORS
        self._map_offsets = (dx, dy, sector)

    def __call__(self, pos: np.ndarray, vel: np.ndarray, targets: np.ndarray | None,
                 scan=None, maps=None) -> np.ndarray:
        n = len(pos)
        obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        obs[:, 0:2] = vel / self.max_speed

        if targets is not None:
            rel = np.nan_to_num(targets - pos)
            d = np.linalg.norm(rel, axis=1, keepdims=True)
            v = rel / TARGET_SCALE
            vn = np.linalg.norm(v, axis=1, keepdims=True)
            obs[:, 2:4] = v / np.maximum(vn, 1.0)
            obs[:, 4] = np.tanh(d[:, 0] / TARGET_SCALE)
            obs[np.isnan(targets[:, 0]), 2:5] = 0.0

        if n > 1:
            k = min(K_NEIGHBORS + 1, n)
            dist, idx = cKDTree(pos).query(pos, k=k)
            dist, idx = dist[:, 1:], idx[:, 1:]                     # drop self
            near = dist <= self.perception
            relp = (pos[idx] - pos[:, None]) / self.perception
            relv = (vel[idx] - vel[:, None]) / self.max_speed
            block = np.concatenate([relp, relv, near[..., None]], axis=2) * near[..., None]
            obs[:, 5:5 + 5 * block.shape[1]] = block.reshape(n, -1)

        rng_max = scan.max_range if scan is not None else 20.0
        if scan is not None:
            rays = scan.dist.shape[1]
            per = max(1, rays // LIDAR_SECTORS)
            sec = scan.dist[:, :per * LIDAR_SECTORS].reshape(n, LIDAR_SECTORS, per).min(axis=2)
        else:
            sec = np.full((n, LIDAR_SECTORS), rng_max)
        if self.world_size is not None:
            # distance along each sector's centre direction to the arena edge
            d = self._sector_dirs[None]                                    # (1, S, 2)
            p = pos[:, None, :]                                            # (N, 1, 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(d > 0, (self.world_size - p) / d, np.where(d < 0, -p / d, np.inf))
            sec = np.minimum(sec, np.clip(t.min(axis=2), 0.0, rng_max))
        obs[:, 35:47] = sec / rng_max

        if maps is not None:
            if self._map_offsets is None:
                self._build_map_offsets(maps)
            dx, dy, sector = self._map_offsets
            cx = np.clip((pos[:, 0] // maps.cell).astype(int)[:, None] + dx[None], 0, maps.gw - 1)
            cy = np.clip((pos[:, 1] // maps.cell).astype(int)[:, None] + dy[None], 0, maps.gh - 1)
            unknown = np.abs(maps.L[np.arange(n)[:, None], cy * maps.gw + cx]) < maps.cfg.known_threshold
            counts = np.bincount(sector, minlength=MAP_SECTORS)
            frac = np.stack([unknown[:, sector == s].sum(1) for s in range(MAP_SECTORS)], 1) / counts
            obs[:, 47:55] = frac
        return obs
