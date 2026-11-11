"""Grid search: cost maps, A*, and a goal-rooted Dijkstra cost field.

All planners work on a drone's OWN occupancy map (never the true world).

Cost map (per cell, metres-equivalent multiplier)
  * known occupied cell          -> blocked
  * within the inflation radius  -> expensive (drone body would scrape)
  * near obstacles               -> a little more expensive (keeps paths centred)
  * unknown cell                 -> slightly more than known free
    ("optimistic" / free-space assumption: plan through the unknown,
     replan when it turns out to be a wall)
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from swarm.config import PlanningConfig

SQRT2 = math.sqrt(2.0)
# 8-connected moves: (dy, dx, length in cells)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]
BLOCKED = np.inf


def cost_map(L: np.ndarray, gh: int, gw: int, cfg: PlanningConfig, known_thr: float,
             inflate_cells: int) -> np.ndarray:
    """Per-cell traversal cost (gh, gw) from one drone's log-odds map (C,)."""
    grid = L.reshape(gh, gw)
    occupied = grid >= known_thr
    unknown = np.abs(grid) < known_thr
    cost = np.ones((gh, gw))
    cost[unknown] += cfg.unknown_cost
    if occupied.any():
        dist = ndimage.distance_transform_edt(~occupied)          # cells to nearest obstacle
        cost += np.where(dist <= inflate_cells, cfg.inflation_cost, 0.0)
        near = np.clip(1.0 - (dist - inflate_cells) / cfg.clearance_cells, 0.0, 1.0)
        cost += cfg.clearance_cost * near * (dist > inflate_cells)
    cost[occupied] = BLOCKED
    return cost


def _diagonal_ok(cost: np.ndarray, y: int, x: int, dy: int, dx: int) -> bool:
    """No corner cutting: both orthogonal neighbours of a diagonal move must be passable."""
    return np.isfinite(cost[y + dy, x]) and np.isfinite(cost[y, x + dx])


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Classic A* on an 8-connected grid with the octile-distance heuristic.

    Edge cost = move length x mean cost of the two cells. The heuristic uses
    the minimum possible cell cost (1), so it never overestimates -> optimal.
    Returns a list of (y, x) cells from start to goal, or None if unreachable.
    """
    gh, gw = cost.shape
    if not (np.isfinite(cost[goal]) and np.isfinite(cost[start])):
        return None
    gy, gx = goal

    def h(y, x):
        dy, dx = abs(y - gy), abs(x - gx)
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)

    g = np.full((gh, gw), np.inf)
    parent = -np.ones((gh, gw), dtype=np.int64)
    g[start] = 0.0
    open_heap = [(h(*start), 0.0, start)]
    closed = np.zeros((gh, gw), dtype=bool)
    while open_heap:
        _, gc, (y, x) = heapq.heappop(open_heap)
        if closed[y, x]:
            continue
        if (y, x) == goal:
            break
        closed[y, x] = True
        c_here = cost[y, x]
        for dy, dx, length in MOVES:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < gh and 0 <= nx < gw) or closed[ny, nx]:
                continue
            c_next = cost[ny, nx]
            if not np.isfinite(c_next):
                continue
            if dy and dx and not _diagonal_ok(cost, y, x, dy, dx):
                continue
            ng = gc + length * 0.5 * (c_here + c_next)
            if ng < g[ny, nx]:
                g[ny, nx] = ng
                parent[ny, nx] = y * gw + x
                heapq.heappush(open_heap, (ng + h(ny, nx), ng, (ny, nx)))
    if not np.isfinite(g[goal]):
        return None
    path = [goal]
    cur = goal[0] * gw + goal[1]
    start_flat = start[0] * gw + start[1]
    while cur != start_flat:
        cur = int(parent[cur // gw, cur % gw])
        path.append((cur // gw, cur % gw))
    return path[::-1]


class GridGraph:
    """Fixed 8-connected neighbour table for one grid size.

    Built once; each plan only fills in edge weights, so rebuilding the
    search graph for a new cost map is a handful of vectorised operations.
    """

    def __init__(self, gh: int, gw: int):
        self.gh, self.gw = gh, gw
        n = gh * gw
        ys, xs = np.divmod(np.arange(n), gw)
        self.nbr = np.zeros((n, 8), dtype=np.int64)       # neighbour cell (self if off-grid)
        self.length = np.zeros(8)
        self.side1 = np.zeros((n, 8), dtype=np.int64)     # cells a diagonal squeezes between
        self.side2 = np.zeros((n, 8), dtype=np.int64)
        self.inside = np.zeros((n, 8), dtype=bool)
        for m, (dy, dx, ln) in enumerate(MOVES):
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < gh) & (nx >= 0) & (nx < gw)
            cy, cx = np.clip(ny, 0, gh - 1), np.clip(nx, 0, gw - 1)
            self.nbr[:, m] = np.where(ok, cy * gw + cx, np.arange(n))
            self.side1[:, m] = ys * gw + cx if dx else np.arange(n)
            self.side2[:, m] = cy * gw + xs if dy else np.arange(n)
            self.inside[:, m] = ok
            self.length[m] = ln

    def edge_weights(self, cost: np.ndarray) -> np.ndarray:
        """(C, 8) move costs; inf where the move is impossible."""
        c = cost.reshape(-1)
        ok = (self.inside & np.isfinite(c)[:, None] & np.isfinite(c[self.nbr])
              & np.isfinite(c[self.side1]) & np.isfinite(c[self.side2]))
        w = np.full(self.nbr.shape, np.inf)
        w[ok] = (self.length[None, :] * 0.5 * (c[:, None] + c[self.nbr]))[ok]
        return w

    def cost_to_go(self, cost: np.ndarray, goal: tuple[int, int],
                   weights: np.ndarray | None = None) -> np.ndarray:
        """Cheapest cost from every cell to `goal` (backward Dijkstra), shape (gh, gw).

        This is the quantity D* Lite maintains: because the search is rooted
        at the goal, one search answers "which way?" from ANY position.
        """
        w = self.edge_weights(cost) if weights is None else weights
        ok = np.isfinite(w)
        n = self.gh * self.gw
        # boolean masking walks the (C, 8) table row by row, so entries are
        # already grouped by source cell: build CSR directly (no COO sort)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(ok.sum(axis=1), out=indptr[1:])
        graph = csr_matrix((w[ok], self.nbr[ok], indptr), shape=(n, n))
        # edges are symmetric, so distances FROM the goal equal distances TO it
        field = dijkstra(graph, directed=True, indices=goal[0] * self.gw + goal[1])
        return field.reshape(self.gh, self.gw)

    def next_cell(self, field: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """For every cell, the neighbour on its optimal route to the goal (C,)."""
        total = weights + field.reshape(-1)[self.nbr]
        best = total.argmin(axis=1)
        nxt = self.nbr[np.arange(len(best)), best]
        stuck = ~np.isfinite(total.min(axis=1))
        nxt[stuck] = -1
        return nxt


def descend(field: np.ndarray, nxt: np.ndarray, start: tuple[int, int],
            max_steps: int = 10_000) -> list[tuple[int, int]] | None:
    """Walk the shortest-path tree from `start` to the goal (where field == 0),
    using the per-cell pointers from GridGraph.next_cell."""
    gh, gw = field.shape
    flat_field = field.reshape(-1)
    cur = start[0] * gw + start[1]
    if not np.isfinite(flat_field[cur]):
        return None
    path = [cur]
    for _ in range(max_steps):
        if flat_field[cur] == 0.0:
            break
        cur = int(nxt[cur])
        if cur < 0:
            break
        path.append(cur)
    return [(c // gw, c % gw) for c in path]
