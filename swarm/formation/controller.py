"""Formation-keeping controller.

Reuses the boids safety terms (anticipatory separation, wall steering, APF)
and replaces flocking (alignment / cohesion / goal) with:
  * leader   -> fly toward its aim at the formation speed
  * follower -> velocity tracking of its slot:
                  v_desired = v_leader + gain * (slot - position)
Followers with no fresh information about the leader fall back to plain
boids flocking until they hear from it again.
ORCA (if enabled) still wraps this controller, so formation keeping can
never override collision avoidance.
"""

from __future__ import annotations

import numpy as np

from swarm.agents.state import SwarmState
from swarm.config import AvoidanceConfig, BoidsConfig, DroneConfig, FormationConfig
from swarm.control.base import StepContext
from swarm.control.boids import BoidsController, _prioritised_sum
from swarm.mathutil import clamp_norm, norms, safe_normalize


class FormationController(BoidsController):
    def __init__(self, weights: BoidsConfig, drone: DroneConfig, avoidance: AvoidanceConfig,
                 formation: FormationConfig, use_apf: bool = False):
        super().__init__(weights, drone, avoidance, use_apf)
        self.f = formation

    def compute(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        ft = ctx.extras.get("formation")
        if ft is None:
            return super().compute(state, ctx)
        pos, vel = state.pos, state.vel
        n = state.n

        flock = super().compute(state, ctx)          # fallback for drones without a slot
        separation = self._separation(state, ctx)
        boundary = self._boundary(state, ctx)
        obstacle = self._obstacle_term(state, ctx)

        track = np.zeros((n, 2))
        # leader: head for its aim; arrive gently at a real goal
        L = ft.leader
        to_aim = ft.leader_aim - pos[L]
        speed = ft.leader_speed
        if ctx.goal is not None:
            speed *= float(np.clip(np.linalg.norm(ctx.goal - pos[L]) / 10.0, 0.0, 1.0))
        track[L] = clamp_norm((safe_normalize(to_aim[None]) * speed - vel[L][None]), self.d.max_accel)[0]

        # followers: track the slot
        has = ~np.isnan(ft.slot[:, 0])
        has[L] = False
        if has.any():
            desired = ft.ref_vel[has] + self.f.gain * (ft.slot[has] - pos[has])
            desired = clamp_norm(desired, self.d.max_speed)
            track[has] = clamp_norm(desired - vel[has], self.d.max_accel)

        w = self.w
        out = _prioritised_sum([
            w.separation_weight * separation,
            self.avoid.apf_weight * obstacle,
            w.boundary_weight * boundary,
            track,
        ], self.d.max_accel)
        lost = ~has
        lost[L] = False
        out[lost] = flock[lost]
        return out
