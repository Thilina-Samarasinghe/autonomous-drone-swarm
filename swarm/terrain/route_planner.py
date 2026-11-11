"""Multi-drone route planning over terrain and threats (pre-flight, at the base).

Every route leaves its start heading east and enters its goal from the west
(`lane` metres), so drones never travel along the crowded start / goal lines.

Four planners, all scored by the SAME RouteCost so they can be compared:

  pso       Particle Swarm Optimisation. A particle is a whole route
            (K free waypoints between start and goal) PLUS a departure delay:
            waiting a few seconds is often a cheaper way to avoid another drone
            than a long detour. Inertia decreases from
            0.9 to 0.4; the swarm's best route wins. Drones are planned one
            after another (priority planning): each later drone's cost includes
            time-conflicts with the routes already chosen, so the swarm's
            routes are deconflicted in space AND time.
  hybrid    PSO whose swarm is seeded with the grid route (and a few
            perturbed copies): starts from a feasible route, then PSO adds what
            the grid can't do (smooth turns, time-deconfliction).
  grid      The Phase 4 cost field (backward Dijkstra) on a 2 m grid whose
            cell cost holds length + threat + slope, with no-fly cells blocked.
            Optimal on its grid, then simplified (Douglas-Peucker), but it
            cannot see turns or other drones.
  straight  start -> goal (baseline).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from swarm.config import TerrainConfig
from swarm.planning.grid import GridGraph, descend
from swarm.terrain.route_cost import RouteCost, RouteScores
from swarm.terrain.terrain import Terrain

PLANNERS = ("pso", "hybrid", "grid", "straight")
SEEDED = 4          # hybrid: particles initialised from the grid route


@dataclass
class RoutePlan:
    routes: list[np.ndarray]           # one polyline (V, 2) per drone
    delays: list[float]                # departure delay per drone (s)
    scores: list[dict]                 # RouteScores.row() per drone, final (with all conflicts)
    seconds: float                     # planning time
    history: list[np.ndarray]          # PSO: best cost per iteration, per drone


# ----------------------------------------------------------------------- PSO
def _lanes(start: np.ndarray, goal: np.ndarray, lane: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed exit point after the start and entry point before the goal."""
    east = np.array([lane, 0.0])
    return start + east, goal - east


def pso_route(start: np.ndarray, goal: np.ndarray, cost: RouteCost, cfg: TerrainConfig,
              rng: np.random.Generator, seed_route: np.ndarray | None = None
              ) -> tuple[np.ndarray, float, np.ndarray]:
    """Returns (route polyline, departure delay, best-cost history)."""
    K, P = cfg.waypoints, cfg.particles
    exit_pt, entry_pt = _lanes(start, goal, cfg.lane)
    t = np.arange(1, K + 1)[:, None] / (K + 1)
    line = exit_pt + (entry_pt - exit_pt) * t                         # (K, 2)
    span = float(np.linalg.norm(entry_pt - exit_pt))
    lo, hi = cost.lo.copy(), cost.hi.copy()
    # keep waypoints between the start lane and the goal lane, so no route runs
    # along the (crowded) start or goal lines
    lo[0] = max(lo[0], min(exit_pt[0], entry_pt[0]))
    hi[0] = min(hi[0], max(exit_pt[0], entry_pt[0]))
    # half the particles start near the straight line, half spread widely
    sigma = np.where(np.arange(P) < P // 2, 0.12, 0.35)[:, None, None] * span
    X = np.clip(line[None] + rng.normal(0.0, 1.0, (P, K, 2)) * sigma, lo, hi)
    X[0] = line                                                       # one particle = straight line
    if seed_route is not None:
        # K waypoints evenly spaced along the seed route, plus small perturbations
        from swarm.terrain.route_cost import resample
        inner = seed_route[1:-1] if len(seed_route) > 3 else seed_route   # drop the lane points
        pts, _ = resample(inner[None], K + 2)
        X[1] = pts[0, 1:-1]
        for i in range(2, min(SEEDED + 1, P)):
            X[i] = np.clip(pts[0, 1:-1] + rng.normal(0.0, 3.0, (K, 2)), lo, hi)
    # departure delays: most particles start with none, some with a random wait
    D = np.where(rng.random(P) < 0.5, 0.0, rng.uniform(0.0, cfg.max_delay, P))
    D[:SEEDED + 1] = 0.0
    VD = np.zeros(P)
    vmax = 0.08 * (hi - lo)
    V = np.zeros_like(X)

    ends = [np.repeat(p[None, None], 1, 0) for p in (start, exit_pt, entry_pt, goal)]

    def polys(x):
        n = len(x)
        s, e1, e2, g = (np.repeat(p, n, 0) for p in ends)
        return np.concatenate([s, e1, x, e2, g], axis=1)

    c = cost(polys(X), D).total
    pbest, pbest_c, pbest_d = X.copy(), c.copy(), D.copy()
    g = int(c.argmin())
    gbest, gbest_c, gbest_d = X[g].copy(), float(c[g]), float(D[g])
    hist = [gbest_c]
    for it in range(cfg.iterations):
        w = 0.9 - 0.5 * it / max(cfg.iterations - 1, 1)
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = w * V + 1.5 * r1 * (pbest - X) + 1.5 * r2 * (gbest[None] - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lo, hi)
        rd1, rd2 = rng.random(P), rng.random(P)
        VD = np.clip(w * VD + 1.5 * rd1 * (pbest_d - D) + 1.5 * rd2 * (gbest_d - D),
                     -0.15 * cfg.max_delay, 0.15 * cfg.max_delay)
        D = np.clip(D + VD, 0.0, cfg.max_delay)
        c = cost(polys(X), D).total
        better = c < pbest_c
        pbest[better], pbest_c[better], pbest_d[better] = X[better], c[better], D[better]
        g = int(pbest_c.argmin())
        if pbest_c[g] < gbest_c:
            gbest, gbest_c, gbest_d = pbest[g].copy(), float(pbest_c[g]), float(pbest_d[g])
        hist.append(gbest_c)
    return polys(gbest[None])[0], gbest_d, np.array(hist)


# ---------------------------------------------------------------------- grid
def _douglas_peucker(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-9:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(d.argmax())
    if d[i] <= tol:
        return np.array([a, b])
    left = _douglas_peucker(pts[: i + 1], tol)
    right = _douglas_peucker(pts[i:], tol)
    return np.vstack([left[:-1], right])


class GridRoutePlanner:
    def __init__(self, terrain: Terrain, cfg: TerrainConfig, world_size: np.ndarray, margin: float,
                 cell: float = 2.0):
        self.cell = cell
        gw, gh = int(np.ceil(world_size[0] / cell)), int(np.ceil(world_size[1] / cell))
        ys, xs = np.mgrid[0:gh, 0:gw]
        centres = np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell], axis=-1)
        from swarm.terrain.route_cost import CEILING_BUFFER, LETHAL_BUFFER
        risk, _ = terrain.risk(centres)
        _, lethal = terrain.risk(centres, LETHAL_BUFFER)
        ground = terrain.height(centres)
        gy, gx = np.gradient(ground, cell)
        # per-metre cost of a cell: the same weights as RouteCost; climbing is
        # direction-dependent, so the grid charges half the slope either way
        cost = cfg.w_length + cfg.w_threat * risk + cfg.w_climb * 0.5 * np.hypot(gx, gy)
        blocked = lethal | (ground > terrain.ceiling - CEILING_BUFFER)
        blocked |= ((centres < margin) | (centres > np.asarray(world_size) - margin)).any(-1)
        cost[blocked] = np.inf
        self.cost = cost
        self.gh, self.gw = gh, gw
        self.graph = GridGraph(gh, gw)

    def route(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        c = self.cost.copy()
        s = (int(start[1] // self.cell), int(start[0] // self.cell))
        g = (int(goal[1] // self.cell), int(goal[0] // self.cell))
        c[s] = min(c[s], 1.0)
        c[g] = min(c[g], 1.0)
        w = self.graph.edge_weights(c)
        field = self.graph.cost_to_go(c, g, w)
        cells = descend(field, self.graph.next_cell(field, w), s)
        if not cells:
            return np.array([start, goal])
        a = np.array(cells, dtype=float)
        pts = np.stack([(a[:, 1] + 0.5) * self.cell, (a[:, 0] + 0.5) * self.cell], 1)
        pts = np.vstack([start, pts[1:-1], goal]) if len(pts) > 2 else np.array([start, goal])
        return _douglas_peucker(pts, tol=self.cell)

    def route_with_lanes(self, start: np.ndarray, goal: np.ndarray, lane: float) -> np.ndarray:
        exit_pt, entry_pt = _lanes(start, goal, lane)
        mid = self.route(exit_pt, entry_pt)
        return np.vstack([start, mid, goal])


# ------------------------------------------------------------------- driver
def plan_routes(method: str, starts: np.ndarray, goals: np.ndarray, terrain: Terrain,
                cfg: TerrainConfig, world_size: np.ndarray, margin: float,
                rng: np.random.Generator) -> RoutePlan:
    if method not in PLANNERS:
        raise ValueError(f"Unknown route planner '{method}'. Choose from {PLANNERS}")
    t0 = time.perf_counter()
    cost = RouteCost(terrain, cfg, world_size, margin)
    routes, delays, hist = [], [], []
    grid = GridRoutePlanner(terrain, cfg, world_size, margin) if method in ("grid", "hybrid") else None
    for s, g in zip(starts, goals):
        d = 0.0
        if method in ("pso", "hybrid"):
            seed = grid.route_with_lanes(s, g, cfg.lane) if method == "hybrid" else None
            r, d, h = pso_route(s, g, cost, cfg, rng, seed)
            hist.append(h)
        elif method == "grid":
            r = grid.route_with_lanes(s, g, cfg.lane)
        else:
            e1, e2 = _lanes(s, g, cfg.lane)
            r = np.array([s, e1, e2, g])
        routes.append(r)
        delays.append(float(d))
        cost.add_planned(r, d)
    seconds = time.perf_counter() - t0
    # final scores: every route against ALL the others (not just earlier ones)
    scores = []
    for i, r in enumerate(routes):
        judge = RouteCost(terrain, cfg, world_size, margin)
        for j, o in enumerate(routes):
            if j != i:
                judge.add_planned(o, delays[j])
        scores.append(judge(r[None], np.array([delays[i]])).row(0))
    return RoutePlan(routes, delays, scores, seconds, hist)
