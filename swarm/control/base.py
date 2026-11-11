"""Controller interface.

Every behaviour (boids, avoidance, formation, planners, and later the RL
policies) maps the current swarm state to a desired acceleration per drone.
Keeping one interface lets hand-written and learned controllers be swapped
and benchmarked against each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from swarm.agents.sensors import Detections, Scan
from swarm.agents.state import SwarmState
from swarm.sim.neighbors import NeighborPairs
from swarm.sim.world import World


@dataclass
class StepContext:
    """Per-step information shared with controllers."""
    world: World
    neighbors: NeighborPairs            # pairs within perception radius
    goal: np.ndarray | None = None      # optional (2,) shared target point
    dt: float = 0.05
    scan: Scan | None = None            # each drone's lidar scan
    detections: Detections | None = None  # nearest obstacle points from the scan
    targets: np.ndarray | None = None   # (N, 2) per-drone steering targets from the planner
    extras: dict = field(default_factory=dict)


class Controller(Protocol):
    def compute(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        """Return desired acceleration, shape (N, 2)."""
        ...
