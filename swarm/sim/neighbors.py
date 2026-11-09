"""Local neighbour queries using a KD-tree (O(N log N) instead of O(N^2))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class NeighborPairs:
    """Directed pairs (i sees j). Every undirected pair appears twice."""
    i: np.ndarray       # (P,) observer index
    j: np.ndarray       # (P,) neighbour index
    diff: np.ndarray    # (P, 2) pos[i] - pos[j]
    dist: np.ndarray    # (P,)

    def within(self, radius: float) -> "NeighborPairs":
        m = self.dist < radius
        return NeighborPairs(self.i[m], self.j[m], self.diff[m], self.dist[m])

    def undirected(self) -> np.ndarray:
        """(K, 2) array of i<j pairs, for drawing links."""
        m = self.i < self.j
        return np.stack([self.i[m], self.j[m]], axis=1)


def neighbor_pairs(pos: np.ndarray, radius: float) -> NeighborPairs:
    if len(pos) < 2:
        empty = np.empty(0, dtype=int)
        return NeighborPairs(empty, empty, np.empty((0, 2)), np.empty(0))
    tree = cKDTree(pos)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        empty = np.empty(0, dtype=int)
        return NeighborPairs(empty, empty, np.empty((0, 2)), np.empty(0))
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    diff = pos[i] - pos[j]
    dist = np.linalg.norm(diff, axis=1)
    return NeighborPairs(i, j, diff, dist)
