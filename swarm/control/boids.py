"""Reynolds boids (1987) with steering behaviours, vectorised.

Each drone only uses neighbours inside its own perception radius:
  * separation - steer away from drones inside the personal-space radius
                 (weighted by 1/d so near drones dominate)
  * alignment  - match the average heading of neighbours
  * cohesion   - steer toward the local centre of mass
  * boundary   - steer back inside the arena near the walls
  * goal       - optional shared target point (arrive behaviour)
  * obstacle   - (optional) artificial potential field from lidar detections
Steering = desired_velocity - current_velocity, clamped to max_accel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.agents.state import SwarmState
from swarm.config import AvoidanceConfig, BoidsConfig, DroneConfig
from swarm.control.base import StepContext
from swarm.mathutil import clamp_norm, norms, safe_normalize


@dataclass
class BoidsTerms:
    separation: np.ndarray
    alignment: np.ndarray
    cohesion: np.ndarray
    boundary: np.ndarray
    goal: np.ndarray
    obstacle: np.ndarray


class BoidsController:
    LOOKAHEAD = 0.6  # seconds of relative-motion prediction for separation

    def __init__(self, weights: BoidsConfig, drone: DroneConfig,
                 avoidance: AvoidanceConfig | None = None, use_apf: bool = False):
        self.w = weights
        self.d = drone
        self.avoid = avoidance or AvoidanceConfig()
        self.use_apf = use_apf
        self.last_terms: BoidsTerms | None = None

    def _obstacle_term(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        """Artificial potential field (Khatib 1986) built from lidar detections.

        Each detected obstacle point within `apf_influence` pushes the drone
        away with strength (1/d - 1/d0). A tangential component steers the
        drone *around* obstacles it is flying into, which reduces the classic
        APF problem of stopping dead in front of an obstacle.
        """
        n = state.n
        out = np.zeros((n, 2))
        det = ctx.detections
        if not self.use_apf or det is None or len(det.i) == 0:
            return out
        d0 = self.avoid.apf_influence
        det = det.within(d0)
        if len(det.i) == 0:
            return out
        r = self.d.radius
        # repulsive force: 0 at the influence edge, max_accel at contact,
        # growing quadratically so distant obstacles barely matter
        closeness = np.clip((d0 - det.dist) / (d0 - r), 0.0, 1.0)
        strength = self.d.max_accel * closeness ** 2
        away = -det.dir
        vel = state.vel[det.i]
        # tangent: perpendicular to `away`, on the side the drone is already moving;
        # only applied when flying *toward* the obstacle
        tangent = np.stack([-away[:, 1], away[:, 0]], 1)
        side = np.sign((tangent * vel).sum(1))
        side[side == 0] = 1.0
        facing = np.clip((vel * det.dir).sum(1) / max(self.d.cruise_speed, 1e-9), 0.0, 1.0)
        push = (away + tangent * (side * facing)[:, None]) * strength[:, None]
        np.add.at(out, det.i, push)
        return clamp_norm(out, self.d.max_accel)

    def _steer(self, desired_dir: np.ndarray, vel: np.ndarray, speed: float) -> np.ndarray:
        """Reynolds steering toward `desired_dir` at `speed`; zero where no desire."""
        has = norms(desired_dir) > 1e-9
        steer = safe_normalize(desired_dir) * speed - vel
        steer[~has] = 0.0
        return clamp_norm(steer, self.d.max_accel)

    def _separation(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        n = state.n
        pos, vel = state.pos, state.vel
        nb = ctx.neighbors
        # anticipatory separation
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
        return self._steer(sep_vec, vel, self.d.max_speed) * urgency[:, None]


    def _boundary(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        push = ctx.world.boundary_push(state.pos)
        return self._steer(push, state.vel, self.d.max_speed) * norms(push)[:, None].clip(0, 1)

    def compute(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        n = state.n
        pos, vel = state.pos, state.vel
        nb = ctx.neighbors

        separation = self._separation(state, ctx)

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

        boundary = self._boundary(state, ctx)

        # --- goal (arrive: slow down inside 10 m) ---
        goal = np.zeros((n, 2))
        if ctx.goal is not None:
            # head for the planner's waypoint if there is one, else straight at the goal;
            # slow down only when close to the real goal
            aim = ctx.targets if ctx.targets is not None else ctx.goal[None, :]
            dist = norms(ctx.goal[None, :] - pos)
            speed = self.d.cruise_speed * np.clip(dist / 10.0, 0.0, 1.0)
            desired = safe_normalize(aim - pos) * speed[:, None]
            goal = clamp_norm(desired - vel, self.d.max_accel)

        obstacle = self._obstacle_term(state, ctx)

        self.last_terms = BoidsTerms(separation, alignment, cohesion, boundary, goal, obstacle)

        # Prioritised acceleration allocation (Reynolds 1999): safety-critical
        # requests are served first; lower-priority ones only get the budget
        # that is left, so they can never cancel collision avoidance.
        w = self.w
        requests = [
            w.separation_weight * separation,
            self.avoid.apf_weight * obstacle,
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
