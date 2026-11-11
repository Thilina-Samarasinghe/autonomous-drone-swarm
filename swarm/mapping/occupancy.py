"""Per-drone occupancy-grid maps built from lidar (log-odds formulation).

Every drone owns a private grid over the arena. Each cell stores the log-odds
that it is occupied:  0 = unknown, < 0 = free, > 0 = occupied.

  * a beam that passes through a cell is evidence the cell is free
  * a beam that ends in a cell is evidence the cell is occupied
  * values are clamped to +/- l_max so a cell can still change its mind

Map sharing (Phase 3): a drone broadcasts the cells it is confident about
and that changed recently ("dirty" cells). A receiver adopts a cell only
when the sender is clearly MORE confident about it than itself.
"More confident wins" is idempotent, so the same information relayed over
many hops never gets counted twice.
"""

from __future__ import annotations

import math

import numpy as np

from swarm.agents.sensors import Scan
from swarm.config import MappingConfig
from swarm.sim.obstacles import Obstacles

ADOPT_MARGIN = 0.5   # sender must be this much more confident before a cell is adopted


class OccupancyMaps:
    def __init__(self, n: int, world_size: np.ndarray, cfg: MappingConfig):
        self.cfg = cfg
        self.cell = cfg.cell_size
        self.gw = int(math.ceil(world_size[0] / self.cell))
        self.gh = int(math.ceil(world_size[1] / self.cell))
        self.num_cells = self.gw * self.gh
        self.L = np.zeros((n, self.num_cells), dtype=np.float32)
        self.dirty = np.zeros((n, self.num_cells), dtype=bool)
        self.own = np.zeros((n, self.num_cells), dtype=bool)   # observed with own lidar

    @property
    def n(self) -> int:
        return self.L.shape[0]

    # ------------------------------------------------------------ indexing
    def cell_of(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat cell index for points (..., 2) and an in-bounds mask."""
        cx = np.floor(pts[..., 0] / self.cell).astype(np.int64)
        cy = np.floor(pts[..., 1] / self.cell).astype(np.int64)
        inside = (cx >= 0) & (cx < self.gw) & (cy >= 0) & (cy < self.gh)
        return cy * self.gw + cx, inside

    def cell_centres(self) -> np.ndarray:
        ys, xs = np.divmod(np.arange(self.num_cells), self.gw)
        return np.stack([(xs + 0.5) * self.cell, (ys + 0.5) * self.cell], axis=1)

    # ------------------------------------------------------------- sensing
    def integrate_scan(self, pos: np.ndarray, scan: Scan) -> None:
        """Fold one lidar scan per drone into that drone's own map."""
        n, c = self.n, self.num_cells
        step = self.cell * 0.5
        ts = np.arange(0.0, scan.max_range, step)                          # (S,)
        # free space: sample points along each beam, stopping short of the hit
        stop = np.where(scan.hits, scan.dist - step, scan.dist)            # (N, R)
        along = ts[None, None, :] < stop[:, :, None]                       # (N, R, S)
        pts = pos[:, None, None, :] + scan.dirs[None, :, None, :] * ts[None, None, :, None]
        idx, inside = self.cell_of(pts)
        drone = np.broadcast_to(np.arange(n)[:, None, None], idx.shape)
        m = along & inside
        free = np.zeros(n * c, dtype=bool)
        free[drone[m] * c + idx[m]] = True

        # occupied: just past each beam's hit point
        occ = np.zeros(n * c, dtype=bool)
        hit = scan.hits
        if hit.any():
            ends = pos[:, None, :] + scan.dirs[None] * (scan.dist + step * 0.5)[:, :, None]
            eidx, einside = self.cell_of(ends)
            dr = np.broadcast_to(np.arange(n)[:, None], eidx.shape)
            mm = hit & einside
            occ[dr[mm] * c + eidx[mm]] = True
        free &= ~occ

        self.own.reshape(-1)[free | occ] = True
        flat = self.L.reshape(-1)
        dirty = self.dirty.reshape(-1)
        lim = self.cfg.l_max
        for mask, delta in ((free, self.cfg.l_free), (occ, self.cfg.l_occ)):
            k = np.flatnonzero(mask)
            if len(k) == 0:
                continue
            before = flat[k]
            after = np.clip(before + delta, -lim, lim)
            flat[k] = after
            # share a cell once we're confident about it and it actually changed
            dirty[k[(after != before) & (np.abs(after) >= self.cfg.known_threshold)]] = True

    # ------------------------------------------------------------- sharing
    def take_message(self, k: int, max_cells: int, rng: np.random.Generator,
                     refresh_fraction: float = 0.3,
                     own_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Cells drone k puts in one message.

        First the dirty (recently learned) cells, up to `max_cells`. Spare room,
        up to `refresh_fraction` of the budget, is filled with a random sample
        of cells it already knows ("anti-entropy"): without this, a cell whose
        only messages were all lost would never be shared again.
        With `own_only` (single-hop radio) only self-observed cells are sent.
        """
        idx = np.flatnonzero(self.dirty[k])
        if len(idx) > max_cells:
            idx = rng.choice(idx, max_cells, replace=False)
        self.dirty[k, idx] = False
        spare = min(max_cells - len(idx), int(max_cells * refresh_fraction))
        if spare > 0:
            pool = np.abs(self.L[k]) >= self.cfg.known_threshold
            if own_only:
                pool &= self.own[k]
            known = np.flatnonzero(pool)
            if len(known):
                extra = rng.choice(known, min(spare, len(known)), replace=False)
                idx = np.concatenate([idx, extra])
        return idx, self.L[k, idx].copy()

    def merge(self, k: int, idx: np.ndarray, vals: np.ndarray, relay: bool) -> int:
        """Merge received cells (possibly from several messages, possibly with
        duplicates) into drone k's map. Returns how many cells were adopted."""
        if len(idx) == 0:
            return 0
        # for duplicate cells keep the most confident incoming value
        order = np.lexsort((-np.abs(vals), idx))
        idx, vals = idx[order], vals[order]
        first = np.r_[True, idx[1:] != idx[:-1]]
        idx, vals = idx[first], vals[first]
        mine = self.L[k, idx]
        better = np.abs(vals) > np.abs(mine) + ADOPT_MARGIN
        if not better.any():
            return 0
        adopt = idx[better]
        self.L[k, adopt] = vals[better]
        if relay:
            self.dirty[k, adopt] = True
        return int(better.sum())

    # ------------------------------------------------------------ queries
    def known(self) -> np.ndarray:
        return np.abs(self.L) >= self.cfg.known_threshold

    def team_map(self) -> np.ndarray:
        """Most confident value per cell across all drones (what the swarm knows)."""
        pick = np.abs(self.L).argmax(axis=0)
        return self.L[pick, np.arange(self.num_cells)]

    def grid(self, values: np.ndarray) -> np.ndarray:
        """Reshape a flat (C,) map to (gh, gw) with row 0 at y = 0."""
        return values.reshape(self.gh, self.gw)


class GroundTruth:
    """True occupancy of every grid cell, used only for evaluation (never by drones)."""

    def __init__(self, maps: OccupancyMaps, obstacles: Obstacles):
        centres = maps.cell_centres()
        if obstacles.count:
            sd, _ = obstacles.signed_distance(centres)
        else:
            sd = np.full(len(centres), np.inf)
        half_diag = maps.cell * math.sqrt(2) / 2
        self.occupied = sd <= 0.0
        self.free = sd >= half_diag
        # cells cut by an obstacle edge are ambiguous and not scored
        self.scored = self.occupied | self.free

    def coverage(self, values: np.ndarray, threshold: float) -> np.ndarray:
        """Fraction of scored cells that are known, per map row. values: (..., C)."""
        known = np.abs(values) >= threshold
        return (known & self.scored).sum(-1) / max(int(self.scored.sum()), 1)

    def accuracy(self, values: np.ndarray, threshold: float) -> float:
        """Of the known scored cells (all maps), the fraction classified correctly."""
        known = (np.abs(values) >= threshold) & self.scored
        if not known.any():
            return 1.0
        says_occ = values > 0
        correct = (says_occ & self.occupied) | (~says_occ & self.free)
        return float((correct & known).sum() / known.sum())
