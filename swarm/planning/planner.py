"""Per-drone path planning and path following for the whole swarm.

Each drone
  1. plans a route to the shared goal on its OWN occupancy map
     (cost field or A*, see grid.py), treating unknown space as passable
  2. replans when it matters:
       - immediately, if a cell on its remaining path is now known to be occupied
       - when its map's obstacle layout changed (revision counter), at most
         once per `replan_interval`, staggered across drones
       - a full refresh every `refresh_interval` (unknown space shrinks over time)
  3. if it drifts away from its path (pushed by neighbours or ORCA), re-attaches:
     with the cost field that is just a lookup, because the field already
     holds the best way to the goal from EVERY cell
  4. steers toward a "carrot": the farthest point on its path, within
     `lookahead` metres, that it can see in a straight line on its map
     (online string pulling: the jagged grid path becomes a smooth one)

The carrot replaces the raw goal in the boids goal term; ORCA still
guarantees safety on top.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from swarm.config import DroneConfig, MappingConfig, PlanningConfig
from swarm.mapping.occupancy import OccupancyMaps
from swarm.planning.grid import GridGraph, astar, cost_map, descend

ALGORITHMS = ("field", "astar")
STRAY_CELLS = 3.0          # re-attach when this many cells away from the path


@dataclass
class PlanStats:
    plans: int = 0              # full searches
    reattach: int = 0           # cheap path re-extractions from a cached field
    failed: int = 0             # no route found on the drone's map
    blocked_replans: int = 0    # replans triggered by a newly discovered obstacle on the path
    plan_seconds: float = 0.0   # CPU time spent in full searches

    @property
    def mean_ms(self) -> float:
        return 1000.0 * self.plan_seconds / max(self.plans, 1)


class SwarmPlanner:
    def __init__(self, n: int, maps: OccupancyMaps, cfg: PlanningConfig,
                 mapping: MappingConfig, drone: DroneConfig, dt: float, safety_margin: float):
        if cfg.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown planning algorithm '{cfg.algorithm}'. Choose from {ALGORITHMS}")
        self.n = n
        self.cfg = cfg
        self.mapping = mapping
        self.cell = maps.cell
        self.gh, self.gw = maps.gh, maps.gw
        self.graph = GridGraph(self.gh, self.gw)
        self.inflate_cells = max(1, math.ceil((drone.radius + safety_margin) / self.cell))
        self.interval = max(1, round(cfg.replan_interval / dt))
        self.refresh = max(self.interval, round(cfg.refresh_interval / dt))
        # stagger: drone k's first plan happens at step k % interval
        self.next_plan = np.arange(n) % self.interval
        self.next_refresh = self.next_plan.copy()
        self.plan_rev = np.full(n, -1, dtype=np.int64)
        self.paths: list[np.ndarray | None] = [None] * n          # (L, 2) world waypoints
        self.path_cells: list[np.ndarray | None] = [None] * n     # (L,) flat cell indices
        self.next_ptr: list[np.ndarray | None] = [None] * n       # cached field pointers
        self.field: list[np.ndarray | None] = [None] * n
        self.unsafe = np.zeros((n, self.gh * self.gw), dtype=bool)  # inflated obstacles per drone
        self.progress = np.zeros(n, dtype=int)
        self.carrots: np.ndarray | None = None
        self._padded: np.ndarray | None = None
        self._lens = np.zeros(n, dtype=int)
        self._dirty_paths = True
        self.stats = PlanStats()

    # ----------------------------------------------------------------- utils
    def _cell(self, p: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(p[0] // self.cell, 0, self.gw - 1))
        y = int(np.clip(p[1] // self.cell, 0, self.gh - 1))
        return y, x

    def _set_path(self, k: int, cells: list[tuple[int, int]] | None) -> None:
        self.progress[k] = 0
        self._dirty_paths = True
        if not cells:
            self.paths[k] = self.path_cells[k] = None
            return
        a = np.array(cells, dtype=float)
        self.paths[k] = np.stack([(a[:, 1] + 0.5) * self.cell, (a[:, 0] + 0.5) * self.cell], 1)
        self.path_cells[k] = (a[:, 0] * self.gw + a[:, 1]).astype(np.int64)

    # ------------------------------------------------------------------ plan
    def plan(self, k: int, pos: np.ndarray, goal: np.ndarray, maps: OccupancyMaps) -> None:
        t0 = time.perf_counter()
        cost = cost_map(maps.L[k], self.gh, self.gw, self.cfg, self.mapping.known_threshold,
                        self.inflate_cells)
        start, target = self._cell(pos), self._cell(goal)
        # never let map noise trap the drone in its own cell or hide the goal
        cost[start] = min(cost[start], self.cfg.inflation_cost)
        cost[target] = min(cost[target], self.cfg.inflation_cost)
        if self.cfg.algorithm == "astar":
            cells = astar(cost, start, target)
        else:
            w = self.graph.edge_weights(cost)
            field = self.graph.cost_to_go(cost, target, w)
            self.next_ptr[k] = self.graph.next_cell(field, w)
            self.field[k] = field
            cells = descend(field, self.next_ptr[k], start)
        self.stats.plans += 1
        self.stats.plan_seconds += time.perf_counter() - t0
        self.unsafe[k] = (cost >= self.cfg.inflation_cost).reshape(-1)
        self.plan_rev[k] = maps.occ_revision[k]
        if not cells:
            self.stats.failed += 1
        self._set_path(k, cells)

    def _reattach(self, k: int, pos: np.ndarray, goal: np.ndarray, maps: OccupancyMaps) -> None:
        if self.cfg.algorithm == "field" and self.next_ptr[k] is not None:
            self.stats.reattach += 1
            self._set_path(k, descend(self.field[k], self.next_ptr[k], self._cell(pos)))
        else:
            self.plan(k, pos, goal, maps)

    def _blocked(self, k: int, L: np.ndarray) -> bool:
        cells = self.path_cells[k]
        if cells is None:
            return False
        return bool((L[cells[self.progress[k]:]] >= self.mapping.known_threshold).any())

    # ---------------------------------------------------------------- update
    def update(self, step: int, pos: np.ndarray, goal: np.ndarray | None,
               maps: OccupancyMaps, goal_radius: float) -> np.ndarray | None:
        """Plan / replan as needed and return each drone's steering target (N, 2)."""
        if goal is None:
            self.carrots = None
            return None
        dist_goal = np.linalg.norm(pos - goal[None], axis=1)
        active = dist_goal >= goal_radius * 0.5        # drones near the goal just fly in
        check_blocked = step % 5 == 0
        for k in np.flatnonzero(active):
            if step >= self.next_plan[k] and (maps.occ_revision[k] != self.plan_rev[k]
                                              or step >= self.next_refresh[k]
                                              or self.paths[k] is None):
                self.plan(k, pos[k], goal, maps)
                self.next_plan[k] = step + self.interval
                self.next_refresh[k] = step + self.refresh
            elif check_blocked and self._blocked(k, maps.L[k]):
                self.stats.blocked_replans += 1
                self.plan(k, pos[k], goal, maps)
                self.next_plan[k] = step + self.interval
        self.carrots = self._carrots(pos, goal, active, maps)
        return self.carrots

    # --------------------------------------------------------------- carrots
    def _pad(self) -> None:
        lens = np.array([0 if p is None else len(p) for p in self.paths])
        lmax = max(int(lens.max()), 1)
        padded = np.zeros((self.n, lmax, 2))
        for k, p in enumerate(self.paths):
            if p is not None:
                padded[k, :len(p)] = p
                padded[k, len(p):] = p[-1]
        self._padded, self._lens = padded, lens
        self._dirty_paths = False

    def _carrots(self, pos, goal, active, maps) -> np.ndarray:
        out = np.repeat(goal[None], self.n, axis=0)
        if self._dirty_paths:
            self._pad()
        has = active & (self._lens > 0)
        if not has.any():
            return out
        look = max(1, int(round(self.cfg.lookahead / self.cell)))
        ks = np.flatnonzero(has)
        P, lens = self._padded[ks], self._lens[ks]
        last = lens - 1

        # progress: nearest waypoint in a window ahead of the previous progress
        win = np.minimum(self.progress[ks, None] + np.arange(3 * look)[None], last[:, None])
        wpts = P[np.arange(len(ks))[:, None], win]
        d = np.linalg.norm(wpts - pos[ks, None], axis=2)
        i0 = win[np.arange(len(ks)), d.argmin(axis=1)]
        self.progress[ks] = i0

        # drifted far from the path? re-attach (cheap with a cached field)
        strayed = d.min(axis=1) > STRAY_CELLS * self.cell
        if strayed.any():
            for k in ks[strayed]:
                self._reattach(int(k), pos[k], goal, maps)
            self._pad()
            P, lens = self._padded[ks], self._lens[ks]
            last = lens - 1
            i0 = np.where(strayed, 0, i0)
            self.progress[ks] = i0

        # string pulling: farthest visible waypoint within the lookahead
        cidx = np.minimum(i0[:, None] + np.arange(look + 1)[None], last[:, None])   # (M, K)
        cand = P[np.arange(len(ks))[:, None], cidx]                                 # (M, K, 2)
        s = np.linspace(0.0, 1.0, 24)[None, None, :, None]
        p0 = pos[ks][:, None, None, :]
        pts = p0 + (cand[:, :, None, :] - p0) * s                                   # (M, K, S, 2)
        near = np.linalg.norm(pts - p0, axis=3) < 1.5 * self.cell
        cx = np.clip((pts[..., 0] // self.cell).astype(int), 0, self.gw - 1)
        cy = np.clip((pts[..., 1] // self.cell).astype(int), 0, self.gh - 1)
        hit = self.unsafe[ks[:, None, None], cy * self.gw + cx] & ~near
        clear = ~hit.any(axis=2)
        clear[:, 0] = True
        first_blocked = np.where((~clear).any(axis=1), (~clear).argmax(axis=1), clear.shape[1])
        pick = np.maximum(first_blocked - 1, 0)
        out[ks] = cand[np.arange(len(ks)), pick]
        return out
