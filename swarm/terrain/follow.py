"""Fly pre-planned routes: each drone steers at a point `lookahead` metres
ahead of its closest point on its own route (pure pursuit), slows down at
the goal, and records what it actually experienced (threat, climb, arrival)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.terrain.route_cost import resample
from swarm.terrain.terrain import Terrain

STEP = 1.0              # metres between route samples
ARRIVE = 3.0            # metres: drone has arrived


@dataclass
class FlightStats:
    threat: np.ndarray           # (N,) integral of risk along the flown path
    lethal_time: np.ndarray      # (N,) seconds spent inside a lethal core
    high_time: np.ndarray        # (N,) seconds over ground above the ceiling
    climb: np.ndarray            # (N,) metres climbed (ground under the drone)
    arrival: np.ndarray          # (N,) arrival time (nan = not yet)
    close_pair_seconds: float = 0.0   # en route, pairs of drones closer than `separation`


class RouteFollower:
    def __init__(self, routes: list[np.ndarray], terrain: Terrain, speed: float, lookahead: float = 8.0,
                 delays: list[float] | None = None, separation: float = 6.0, endpoint_zone: float = 8.0):
        self.terrain = terrain
        self.separation = separation
        self.endpoint_zone = endpoint_zone
        self.delays = np.zeros(len(routes)) if delays is None else np.asarray(delays, dtype=float)
        self.speed = speed
        self.lookahead = lookahead
        self.paths = []
        for r in routes:
            length = float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum())
            pts, _ = resample(r[None], max(2, int(length / STEP) + 1))
            self.paths.append(pts[0])
        n = len(routes)
        self.goals = np.array([p[-1] for p in self.paths])
        self.starts = np.array([p[0] for p in self.paths])
        self.progress = np.zeros(n, dtype=int)
        self.stats = FlightStats(np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, np.nan))
        self._last_ground: np.ndarray | None = None

    def desired_velocity(self, pos: np.ndarray, time: float = float("inf")) -> np.ndarray:
        n = len(self.paths)
        out = np.zeros((n, 2))
        ahead = int(self.lookahead / STEP)
        for k, path in enumerate(self.paths):
            if time < self.delays[k]:
                continue                                    # waiting for its departure slot
            lo = self.progress[k]
            hi = min(len(path), lo + 4 * ahead)
            i = lo + int(np.argmin(np.linalg.norm(path[lo:hi] - pos[k], axis=1)))
            self.progress[k] = i
            target = path[min(i + ahead, len(path) - 1)]
            to = target - pos[k]
            d = np.linalg.norm(to)
            dist_goal = np.linalg.norm(self.goals[k] - pos[k])
            speed = self.speed * np.clip(dist_goal / 6.0, 0.0, 1.0)
            out[k] = to / d * speed if d > 1e-6 else 0.0
        return out

    def record(self, time: float, pos: np.ndarray, vel: np.ndarray, dt: float) -> None:
        risk, lethal = self.terrain.risk(pos)
        speed = np.linalg.norm(vel, axis=1)
        self.stats.threat += risk * speed * dt
        self.stats.lethal_time += lethal * dt
        ground = self.terrain.height(pos)
        self.stats.high_time += (ground > self.terrain.ceiling) * dt
        if self._last_ground is not None:
            self.stats.climb += np.clip(ground - self._last_ground, 0.0, None)
        self._last_ground = ground
        arrived = (np.linalg.norm(pos - self.goals, axis=1) < ARRIVE) & np.isnan(self.stats.arrival)
        self.stats.arrival[arrived] = time
        # close encounters en route (the thing planned deconfliction is meant to prevent);
        # crowds around a drone's own start or goal are excluded, as in planning
        en_route = ((np.linalg.norm(pos - self.starts, axis=1) > self.endpoint_zone)
                    & (np.linalg.norm(pos - self.goals, axis=1) > self.endpoint_zone))
        p = pos[en_route]
        if len(p) > 1:
            d = np.linalg.norm(p[:, None] - p[None], axis=2)
            close = int(((d < self.separation) & np.triu(np.ones_like(d, dtype=bool), 1)).sum())
            self.stats.close_pair_seconds += close * dt
