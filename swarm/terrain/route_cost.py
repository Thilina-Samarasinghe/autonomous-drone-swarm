"""Cost of a flight route over terrain and threats, evaluated for many routes at once.

A route is a polyline: start, K free waypoints, goal. It is resampled to a
FIXED number of points (SAMPLES, < 1 m apart for any route in the arena) so
that a route's score never depends on which other routes it is evaluated
alongside, and scored with

  length     metres flown
  threat     integral of risk along the route   (risk = sum of (1 - d/R)^2)
  lethal     metres flown inside a threat's lethal core       (forbidden)
  too_high   metres flown over ground above the ceiling        (forbidden)
             (forbidden terms are enforced with a safety buffer: cores
             LETHAL_BUFFER m bigger, ceiling CEILING_BUFFER m lower, because a
             drone in flight may be pushed a little off its route; the scores
             report the TRUE violations)
  climb      total metres of climbing (flying `clearance` above the ground)
  turn       sum of squared turn angles beyond TURN_FREE at the waypoints
  outside    metres flown outside the arena's safe area        (forbidden)
  conflict   for each already-planned drone: how far inside the separation
             distance the two drones get AT THE SAME TIME, flying at `speed`
             after their departure delays
             (not counted while BOTH drones are within ENDPOINT_ZONE of their
             own start or goal: drones take off side by side, and ORCA handles
             that crowd. Flying past a drone already parked at its goal does
             count.)

  delay      seconds the drone waits at its start before taking off

Total = weighted sum. "Forbidden" terms carry very large weights, so a
feasible route always beats an infeasible one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.config import TerrainConfig
from swarm.terrain.terrain import Terrain

SAMPLES = 400                   # points per route, whatever the route or the batch
TURN_FREE = np.radians(30.0)    # turns gentler than this are free
FORBIDDEN = 1e4                 # weight of forbidden terms (per metre)
LETHAL_BUFFER = 3.0             # plan as if lethal cores were this much bigger (m)
CEILING_BUFFER = 5.0            # ... and the ceiling this much lower (m)
ENDPOINT_ZONE = 8.0             # m around a drone's own start / goal where conflicts don't count


@dataclass
class RouteScores:
    total: np.ndarray
    length: np.ndarray
    threat: np.ndarray
    lethal: np.ndarray
    too_high: np.ndarray
    climb: np.ndarray
    turn: np.ndarray
    outside: np.ndarray
    conflict: np.ndarray
    delay: np.ndarray

    def row(self, i: int) -> dict:
        return {k: float(getattr(self, k)[i]) for k in self.__dataclass_fields__}


def resample(poly: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample polylines (P, V, 2) to m points evenly spaced by arc length.
    Returns (points (P, m, 2), total length (P,))."""
    seg = np.diff(poly, axis=1)
    seg_len = np.linalg.norm(seg, axis=2)                          # (P, V-1)
    cum = np.concatenate([np.zeros((len(poly), 1)), np.cumsum(seg_len, axis=1)], axis=1)
    total = cum[:, -1]
    arc = np.linspace(0.0, 1.0, m)[None] * total[:, None]           # (P, m)
    idx = np.clip((cum[:, None, :] <= arc[:, :, None]).sum(-1) - 1, 0, seg.shape[1] - 1)
    rows = np.arange(len(poly))[:, None]
    frac = (arc - cum[rows, idx]) / np.maximum(seg_len[rows, idx], 1e-9)
    pts = poly[rows, idx] + seg[rows, idx] * np.clip(frac, 0.0, 1.0)[..., None]
    return pts, total


def positions_at_times(poly: np.ndarray, times: np.ndarray, speed: float,
                       delay: np.ndarray | float = 0.0) -> np.ndarray:
    """Where a drone flying polyline(s) (P, V, 2) at `speed` is at `times` (J,):
    (P, J, 2). It waits at its start for `delay` seconds (scalar or (P,)) and at
    the goal once it gets there."""
    seg = np.diff(poly, axis=1)
    seg_len = np.linalg.norm(seg, axis=2)
    cum = np.concatenate([np.zeros((len(poly), 1)), np.cumsum(seg_len, axis=1)], axis=1)
    delay = np.broadcast_to(np.asarray(delay, dtype=float), (len(poly),))[:, None]
    arc = np.minimum(np.clip(times[None] - delay, 0.0, None) * speed, cum[:, -1:])   # (P, J)
    idx = np.clip((cum[:, None, :] <= arc[:, :, None]).sum(-1) - 1, 0, seg.shape[1] - 1)
    rows = np.arange(len(poly))[:, None]
    frac = (arc - cum[rows, idx]) / np.maximum(seg_len[rows, idx], 1e-9)
    return poly[rows, idx] + seg[rows, idx] * np.clip(frac, 0.0, 1.0)[..., None]


class RouteCost:
    def __init__(self, terrain: Terrain, cfg: TerrainConfig, world_size: np.ndarray, margin: float):
        self.t = terrain
        self.cfg = cfg
        self.tab = terrain.tables(np.asarray(world_size, dtype=float), LETHAL_BUFFER)
        self.lo = np.array([margin, margin])
        self.hi = np.asarray(world_size, dtype=float) - margin
        self.others: list[np.ndarray] = []          # polylines of drones already planned
        self.other_delays: list[float] = []

    def add_planned(self, poly: np.ndarray, delay: float = 0.0) -> None:
        self.others.append(np.asarray(poly, dtype=float))
        self.other_delays.append(float(delay))

    def __call__(self, poly: np.ndarray, delay: np.ndarray | None = None) -> RouteScores:
        """Score routes (P, V, 2) with departure delays (P,) (default 0)."""
        cfg = self.cfg
        P = len(poly)
        delay = np.zeros(P) if delay is None else np.asarray(delay, dtype=float)
        m = SAMPLES
        pts, length = resample(poly, m)
        ds = (length / (m - 1))[:, None]                             # (P, 1)

        iy, ix = self.tab.index(pts)
        risk, lethal, lethal_buf = self.tab.risk[iy, ix], self.tab.lethal[iy, ix], self.tab.lethal_buf[iy, ix]
        threat = (risk * ds).sum(1)
        lethal_m = (lethal * ds).sum(1)
        ground = self.tab.height[iy, ix]
        too_high = ((ground > self.t.ceiling) * ds).sum(1)
        forbidden = ((lethal_buf | (ground > self.t.ceiling - CEILING_BUFFER)) * ds).sum(1)
        climb = np.clip(np.diff(ground, axis=1), 0.0, None).sum(1)

        out = ((pts < self.lo) | (pts > self.hi)).any(-1)
        outside = (out * ds).sum(1)

        d = np.diff(poly, axis=1)
        n = np.linalg.norm(d, axis=2)
        u = d / np.maximum(n, 1e-9)[..., None]
        cosang = np.clip((u[:, 1:] * u[:, :-1]).sum(-1), -1.0, 1.0)
        turn = (np.clip(np.arccos(cosang) - TURN_FREE, 0.0, None) ** 2).sum(1)

        conflict = np.zeros(P)
        if self.others:
            horizon = max(length.max() / cfg.speed + delay.max(),
                          max(np.linalg.norm(np.diff(o, axis=0), axis=1).sum() / cfg.speed + d
                              for o, d in zip(self.others, self.other_delays)))
            times = np.arange(0.0, horizon + 1.0, 0.5)
            mine = positions_at_times(poly, times, cfg.speed, delay)                # (P, J, 2)
            others = np.stack([np.stack([o[0], o[-1]]) for o in self.others])      # (Q, 2, 2)
            theirs = np.concatenate([positions_at_times(o[None], times, cfg.speed, d)
                                     for o, d in zip(self.others, self.other_delays)])  # (Q, J, 2)
            mine_free = ((np.linalg.norm(mine - poly[:, :1], axis=2) > ENDPOINT_ZONE)
                         & (np.linalg.norm(mine - poly[:, -1:], axis=2) > ENDPOINT_ZONE))      # (P, J)
            theirs_free = ((np.linalg.norm(theirs - others[:, :1], axis=2) > ENDPOINT_ZONE)
                           & (np.linalg.norm(theirs - others[:, 1:], axis=2) > ENDPOINT_ZONE))  # (Q, J)
            gap = np.linalg.norm(mine[:, None] - theirs[None], axis=3)              # (P, Q, J)
            counted = mine_free[:, None] | theirs_free[None]
            conflict = (np.clip(cfg.separation - gap, 0.0, None) * counted).sum((1, 2))

        total = (cfg.w_length * length + cfg.w_threat * threat + cfg.w_climb * climb
                 + cfg.w_turn * turn + cfg.w_conflict * conflict + cfg.w_delay * delay
                 + FORBIDDEN * (forbidden + outside))
        return RouteScores(total, length, threat, lethal_m, too_high, climb, turn, outside, conflict, delay)
