"""Static obstacles: circles and axis-aligned boxes, with vectorised geometry.

Provides
  * signed_distance - distance from each drone centre to the nearest obstacle
                      surface (negative inside) plus the outward surface normal
  * raycast         - lidar-style ray intersection distances
  * resolve         - hard physics: push penetrating drones back out
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INF = np.inf


@dataclass
class Obstacles:
    circles: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # (C, 3): cx, cy, r
    boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))    # (B, 4): x0, y0, x1, y1

    def __post_init__(self):
        self.circles = np.asarray(self.circles, dtype=float).reshape(-1, 3)
        self.boxes = np.asarray(self.boxes, dtype=float).reshape(-1, 4)

    @property
    def count(self) -> int:
        return len(self.circles) + len(self.boxes)

    # ---------------------------------------------------------- distances
    def signed_distance(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the nearest obstacle surface, shape (N,), and the
        outward unit normal at that point, shape (N, 2). inf where no obstacles."""
        n = len(pos)
        best = np.full(n, INF)
        normal = np.zeros((n, 2))

        if len(self.circles):
            diff = pos[:, None, :] - self.circles[None, :, :2]          # (N, C, 2)
            centre_d = np.linalg.norm(diff, axis=2)                      # (N, C)
            d = centre_d - self.circles[None, :, 2]
            k = d.argmin(axis=1)
            rows = np.arange(n)
            dk = d[rows, k]
            nk = diff[rows, k] / np.maximum(centre_d[rows, k], 1e-9)[:, None]
            better = dk < best
            best[better], normal[better] = dk[better], nk[better]

        if len(self.boxes):
            lo, hi = self.boxes[:, :2], self.boxes[:, 2:]
            centre, half = (lo + hi) / 2, (hi - lo) / 2
            rel = pos[:, None, :] - centre[None]                         # (N, B, 2)
            q = np.abs(rel) - half[None]                                 # (N, B, 2)
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=2)
            inside = np.minimum(q.max(axis=2), 0.0)
            d = outside + inside                                         # (N, B)
            k = d.argmin(axis=1)
            rows = np.arange(n)
            dk = d[rows, k]
            qk, relk = q[rows, k], rel[rows, k]
            sign = np.where(relk >= 0, 1.0, -1.0)
            out_n = np.maximum(qk, 0.0) * sign
            out_len = np.linalg.norm(out_n, axis=1)
            # inside: push out along the axis of least penetration
            axis = qk.argmax(axis=1)
            in_n = np.zeros((n, 2))
            in_n[rows, axis] = sign[rows, axis]
            nk = np.where((out_len > 1e-9)[:, None], out_n / np.maximum(out_len, 1e-9)[:, None], in_n)
            better = dk < best
            best[better], normal[better] = dk[better], nk[better]

        return best, normal

    # -------------------------------------------------------------- rays
    def raycast(self, origins: np.ndarray, dirs: np.ndarray, max_range: float) -> np.ndarray:
        """Distance along each ray to the first hit, shape (N, R).
        `dirs` are unit vectors (R, 2). Misses return `max_range`."""
        n, r = len(origins), len(dirs)
        out = np.full((n, r), float(max_range))
        dx, dy = dirs[None, :, 0], dirs[None, :, 1]                     # (1, R)

        if len(self.circles):
            # only test obstacles that are within sensor range of each drone
            centre_d = np.linalg.norm(origins[:, None] - self.circles[None, :, :2], axis=2)
            di, ci = np.nonzero(centre_d - self.circles[None, :, 2] < max_range)
            if len(di):
                fx = (origins[di, 0] - self.circles[ci, 0])[:, None]     # (P, 1)
                fy = (origins[di, 1] - self.circles[ci, 1])[:, None]
                rad = self.circles[ci, 2][:, None]
                b = fx * dx + fy * dy                                    # (P, R)
                c = fx * fx + fy * fy - rad * rad                        # (P, 1)
                disc = b * b - c
                t = -b - np.sqrt(np.maximum(disc, 0.0))
                t = np.where(c <= 0, 0.0, t)                             # origin inside
                t = np.where((disc >= 0) & (t >= 0), t, INF)
                np.minimum.at(out, di, t)

        if len(self.boxes):
            lo_all, hi_all = self.boxes[:, :2], self.boxes[:, 2:]
            gap = np.linalg.norm(np.maximum(np.maximum(lo_all[None] - origins[:, None],
                                                       origins[:, None] - hi_all[None]), 0.0), axis=2)
            di, bi = np.nonzero(gap < max_range)
            if len(di):
                o = origins[di][:, None, :]                              # (P,1,2)
                lo, hi = lo_all[bi][:, None, :], hi_all[bi][:, None, :]
                d = dirs[None]                                           # (1,R,2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv = 1.0 / d
                    t1 = (lo - o) * inv
                    t2 = (hi - o) * inv
                # rays parallel to a slab: inside -> (-inf, inf) never limits;
                # outside -> (inf, inf) so the entry time is inf and nothing is hit
                par = d == 0
                in_slab = (o >= lo) & (o <= hi)
                t1 = np.where(par, np.where(in_slab, -INF, INF), t1)
                t2 = np.where(par, INF, t2)
                tmin = np.minimum(t1, t2).max(-1)                        # (P, R)
                tmax = np.maximum(t1, t2).min(-1)
                hit = tmax >= np.maximum(tmin, 0.0)
                t = np.where(hit, np.maximum(tmin, 0.0), INF)
                np.minimum.at(out, di, t)

        return np.minimum(out, max_range)

    # ------------------------------------------------------------ physics
    def resolve(self, pos: np.ndarray, vel: np.ndarray, radius: float) -> np.ndarray:
        """Push drones that penetrate an obstacle back to its surface and
        cancel their inward velocity (in place). Returns the hit mask (N,)."""
        if self.count == 0 or len(pos) == 0:
            return np.zeros(len(pos), dtype=bool)
        sd, nrm = self.signed_distance(pos)
        hit = sd < radius
        if hit.any():
            pos[hit] += nrm[hit] * (radius - sd[hit])[:, None]
            vn = (vel[hit] * nrm[hit]).sum(1)
            inward = vn < 0
            idx = np.flatnonzero(hit)[inward]
            vel[idx] -= nrm[idx] * vn[inward][:, None]
        return hit
