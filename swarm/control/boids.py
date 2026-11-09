"""Reynolds boids (1987) with steering behaviours, vectorised.

Each drone only uses neighbours inside its own perception radius:
  * separation - steer away from drones inside the personal-space radius
                 (weighted by 1/d so near drones dominate)
  * alignment  - match the average heading of neighbours
  * cohesion   - steer toward the local centre of mass
  * boundary   - steer back inside the arena near the walls
  * goal       - optional shared target point (arrive behaviour)
Steering = desired_velocity - current_velocity, clamped to max_accel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.agents.state import SwarmState
from swarm.config import BoidsConfig, DroneConfig
from swarm.control.base import StepContext
from swarm.mathutil import clamp_norm, norms, safe_normalize


@dataclass
class BoidsTerms:
    separation: np.ndarray
    alignment: np.ndarray
    cohesion: np.ndarray
    boundary: np.ndarray
    goal: np.ndarray


class BoidsController:
    LOOKAHEAD = 0.6  # seconds of relative-motion prediction for separation

    def __init__(self, weights: BoidsConfig, drone: DroneConfig):
        self.w = weights
        self.d = drone
        self.last_terms: BoidsTerms | None = None

    def _steer(self, desired_dir: np.ndarray, vel: np.ndarray, speed: float) -> np.ndarray:
        """Reynolds steering toward `desired_dir` at `speed`; zero where no desire."""
        has = norms(desired_dir) > 1e-9
        steer = safe_normalize(desired_dir) * speed - vel
        steer[~has] = 0.0
        return clamp_norm(steer, self.d.max_accel)

    def compute(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        n = state.n
        pos, vel = state.pos, state.vel
        nb = ctx.neighbors

        # --- separation (anticipatory) ---
        # Each drone also looks LOOKAHEAD seconds ahead using the relative
        # velocity, and reacts to whichever is closer: now or predicted.
        # This handles the compression that happens when a flock turns.
        sep_vec = np.zeros((n, 2))
        if len(nb.i):
            rel_v = vel[nb.i] - vel[nb.j]
            pred = nb.diff + rel_v * self.LOOKAHEAD
            pred_dist = norms(pred)
            use_pred = pred_dist < nb.dist
            diff = np.where(use_pred[:, None], pred, nb.diff)
            dist = np.where(use_pred, pred_dist, nb.dist)
            # predicted overlap: fall back to the current offset for direction
            degenerate = dist < 1e-3
            diff[degenerate] = nb.diff[degenerate]
            dist = np.maximum(dist, 1e-3)
            m = dist < self.d.separation_radius
            if m.any():
                # away from neighbour, weight 1/d (stronger when closer)
                push = diff[m] / dist[m, None] ** 2
                np.add.at(sep_vec, nb.i[m], push)
        # Urgency: 0 at the personal-space edge, 1 at contact distance.
        # d_eff = 1/|sum(1/d)| behaves like the distance to the nearest drone.
        d_eff = 1.0 / np.maximum(norms(sep_vec), 1e-9)
        contact = 2 * self.d.radius
        span = max(self.d.separation_radius - contact, 1e-6)
        urgency = np.sqrt(np.clip((self.d.separation_radius - d_eff) / span, 0.0, 1.0))
        separation = self._steer(sep_vec, vel, self.d.max_speed) * urgency[:, None]

        # --- alignment & cohesion (all perceived neighbours) ---
        counts = np.bincount(nb.i, minlength=n).astype(float)
        has_nb = counts > 0
        vel_sum = np.zeros((n, 2))
        pos_sum = np.zeros((n, 2))
        if len(nb.i):
            np.add.at(vel_sum, nb.i, vel[nb.j])
            np.add.at(pos_sum, nb.i, pos[nb.j])
        denom = np.maximum(counts, 1.0)[:, None]
        mean_vel = vel_sum / denom
        centroid = pos_sum / denom

        align_dir = np.where(has_nb[:, None], mean_vel, 0.0)
        alignment = self._steer(align_dir, vel, self.d.cruise_speed)

        # Cohesion is a pull toward the local centre that grows with distance.
        # (Not "desired - vel": that form brakes drones already near the centre.)
        offset = np.where(has_nb[:, None], centroid - pos, 0.0)
        cohesion = offset / self.d.perception_radius * self.d.max_accel

        # Speed keeping: hold cruise speed along the current heading.
        cruise = self._steer(vel, vel, self.d.cruise_speed)

        # --- boundary ---
        push = ctx.world.boundary_push(pos)
        boundary = self._steer(push, vel, self.d.max_speed) * norms(push)[:, None].clip(0, 1)

        # --- goal (arrive: slow down inside 10 m) ---
        goal = np.zeros((n, 2))
        if ctx.goal is not None:
            to_goal = ctx.goal[None, :] - pos
            dist = norms(to_goal)
            speed = self.d.cruise_speed * np.clip(dist / 10.0, 0.0, 1.0)
            desired = safe_normalize(to_goal) * speed[:, None]
            goal = clamp_norm(desired - vel, self.d.max_accel)

        self.last_terms = BoidsTerms(separation, alignment, cohesion, boundary, goal)

        # Prioritised acceleration allocation (Reynolds 1999): safety-critical
        # requests are served first; lower-priority ones only get the budget
        # that is left, so they can never cancel collision avoidance.
        w = self.w
        requests = [
            w.separation_weight * separation,
            w.boundary_weight * boundary,
            w.goal_weight * goal,
            w.alignment_weight * alignment + w.cohesion_weight * cohesion,
            0.5 * cruise,
        ]
        return _prioritised_sum(requests, self.d.max_accel)


def _prioritised_sum(requests: list[np.ndarray], budget: float) -> np.ndarray:
    total = np.zeros_like(requests[0])
    remaining = np.full(len(total), budget)
    for req in requests:
        req = clamp_norm(req, budget)
        mag = norms(req)
        scale = np.where(mag > remaining, remaining / np.maximum(mag, 1e-9), 1.0)
        total += req * scale[:, None]
        remaining = np.maximum(remaining - mag * scale, 0.0)
    return total
