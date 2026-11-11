"""Mission controller: track the velocity the mission asks for.

Keeps the boids safety terms (separation, APF) and replaces flocking with
velocity tracking, because search drones should spread out, not stick
together. ORCA (if on) still wraps it.

The soft arena-edge push comes AFTER tracking here: a search drone may
deliberately work close to the edge, and with the push first a drone in a
slot between an obstacle and the edge could be pinned there (the hard arena
walls still stop it physically).
"""

from __future__ import annotations

import numpy as np

from swarm.agents.state import SwarmState
from swarm.control.base import StepContext
from swarm.control.boids import BoidsController, _prioritised_sum
from swarm.mathutil import clamp_norm


class MissionController(BoidsController):
    def compute(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        desired = ctx.extras.get("mission")
        if desired is None:
            return super().compute(state, ctx)
        track = clamp_norm(clamp_norm(desired, self.d.max_speed) - state.vel, self.d.max_accel)
        w = self.w
        return _prioritised_sum([
            w.separation_weight * self._separation(state, ctx),
            self.avoid.apf_weight * self._obstacle_term(state, ctx),
            track,
            w.boundary_weight * self._boundary(state, ctx),
        ], self.d.max_accel)
