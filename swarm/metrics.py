"""Swarm-level metrics used for the HUD, logs and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from swarm.agents.state import SwarmState
from swarm.mathutil import norms, safe_normalize
from swarm.sim.neighbors import NeighborPairs


@dataclass
class Metrics:
    time: float
    n: int
    min_separation: float   # closest centre-to-centre distance (m)
    collisions: int         # pairs currently overlapping (dist < 2*radius)
    mean_speed: float
    polarization: float     # global: 0 = random headings, 1 = everyone aligned
    local_polarization: float  # mean alignment of each drone with its neighbours
    groups: int             # connected components of the perception graph

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(state: SwarmState, neighbors: NeighborPairs, radius: float, time: float) -> Metrics:
    n = state.n
    if n < 2:
        return Metrics(time, n, float("inf"), 0, float(norms(state.vel).mean()) if n else 0.0, 1.0, 1.0, n)

    tree = cKDTree(state.pos)
    d, _ = tree.query(state.pos, k=2)
    min_sep = float(d[:, 1].min())
    collisions = len(tree.query_pairs(2 * radius))

    speeds = norms(state.vel)
    heading = safe_normalize(state.vel)
    polarization = float(norms(heading.mean(axis=0, keepdims=True))[0])
    local_sum = heading.copy()
    np.add.at(local_sum, neighbors.i, heading[neighbors.j])
    counts = np.bincount(neighbors.i, minlength=n) + 1
    local_polarization = float((norms(local_sum) / counts).mean())

    graph = coo_matrix((np.ones(len(neighbors.i)), (neighbors.i, neighbors.j)), shape=(n, n))
    groups, _ = connected_components(graph, directed=False)

    return Metrics(time, n, min_sep, collisions, float(speeds.mean()), polarization,
                   local_polarization, int(groups))
