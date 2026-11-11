"""ORCA - Optimal Reciprocal Collision Avoidance (van den Berg et al., 2011).

Used as a *safety layer* wrapped around any other controller:

  1. the inner controller (e.g. boids) proposes an acceleration
  2. that becomes a preferred velocity  v_pref = v + a * dt
  3. every drone builds half-plane constraints on its own velocity:
       - one per nearby drone (reciprocal: each takes half the avoidance effort)
       - one per obstacle point its lidar detects (full effort, obstacles don't move)
  4. a small 2D linear program finds the velocity closest to v_pref that
     satisfies all constraints (or violates them least, if infeasible)
  5. the change in velocity is turned back into an acceleration

Each drone solves its own problem from local information only.
The drone-drone constraint geometry is vectorised with NumPy; the per-drone
linear programs are a direct port of the RVO2 library's solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from swarm.agents.state import SwarmState
from swarm.config import AvoidanceConfig, DroneConfig
from swarm.control.base import Controller, StepContext
from swarm.mathutil import clamp_norm

EPS = 1e-5


@dataclass
class OrcaStats:
    constraints: int = 0      # total half-planes built this step
    infeasible: int = 0       # drones whose constraints could not all be met
    adjusted: int = 0         # drones whose velocity ORCA changed noticeably


class OrcaSafetyLayer:
    def __init__(self, inner: Controller, drone: DroneConfig, cfg: AvoidanceConfig):
        self.inner = inner
        self.d = drone
        self.cfg = cfg
        self.stats = OrcaStats()

    # The inner controller may expose extra state (e.g. boids terms).
    def __getattr__(self, item):
        return getattr(self.inner, item)

    def compute(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        dt = ctx.dt
        a_nom = self.inner.compute(state, ctx)
        v_pref = clamp_norm(state.vel + a_nom * dt, self.d.max_speed)

        agent_lines = self._agent_lines(state, ctx, dt)
        obst_lines = self._obstacle_lines(state, ctx, dt)

        n = state.n
        vel = state.vel
        v_new = v_pref.copy()
        stats = OrcaStats()
        vmax = self.d.max_speed
        vp = v_pref.tolist()
        for k in range(n):
            o_lines = obst_lines.get(k, [])
            a_lines = agent_lines.get(k, [])
            if not o_lines and not a_lines:
                continue
            lines = o_lines + a_lines
            stats.constraints += len(lines)
            pref = (vp[k][0], vp[k][1])
            fail, result = _lp2(lines, vmax, pref, False, pref)
            if fail < len(lines):
                stats.infeasible += 1
                result = _lp3(lines, len(o_lines), fail, vmax, result)
            v_new[k] = result

        accel = (v_new - vel) / dt
        stats.adjusted = int((np.linalg.norm(v_new - v_pref, axis=1) > 0.05).sum())
        self.stats = stats
        return clamp_norm(accel, self.d.max_accel)

    # ------------------------------------------------------------------
    def _agent_lines(self, state: SwarmState, ctx: StepContext, dt: float) -> dict[int, list]:
        nb = ctx.neighbors.within(self.cfg.orca_neighbor_dist)
        if len(nb.i) == 0:
            return {}
        # keep the K nearest neighbours of each drone
        order = np.lexsort((nb.dist, nb.i))
        i, j, dist = nb.i[order], nb.j[order], nb.dist[order]
        starts = np.r_[0, np.flatnonzero(np.diff(i)) + 1]
        rank = np.arange(len(i)) - np.repeat(starts, np.diff(np.r_[starts, len(i)]))
        keep = rank < self.cfg.orca_max_neighbors
        i, j = i[keep], j[keep]

        pos, vel = state.pos, state.vel
        rel_pos = pos[j] - pos[i]                     # (P, 2) other - self
        rel_vel = vel[i] - vel[j]
        dist_sq = (rel_pos ** 2).sum(1)
        R = 2 * self.d.radius + self.cfg.safety_margin
        R_sq = R * R
        inv_tau = 1.0 / self.cfg.orca_time_horizon

        direction = np.zeros_like(rel_pos)
        u = np.zeros_like(rel_pos)

        colliding = dist_sq <= R_sq
        # ---- no collision yet
        w = rel_vel - inv_tau * rel_pos
        w_len_sq = (w ** 2).sum(1)
        dot1 = (w * rel_pos).sum(1)
        cutoff = ~colliding & (dot1 < 0) & (dot1 ** 2 > R_sq * w_len_sq)
        legs = ~colliding & ~cutoff

        if cutoff.any():
            wl = np.sqrt(w_len_sq[cutoff])
            unit_w = w[cutoff] / wl[:, None]
            direction[cutoff] = np.stack([unit_w[:, 1], -unit_w[:, 0]], 1)
            u[cutoff] = (R * inv_tau - wl)[:, None] * unit_w

        if legs.any():
            rp, ds = rel_pos[legs], dist_sq[legs]
            leg = np.sqrt(np.maximum(ds - R_sq, 0.0))
            left = (rp[:, 0] * w[legs][:, 1] - rp[:, 1] * w[legs][:, 0]) > 0
            dl = np.stack([rp[:, 0] * leg - rp[:, 1] * R, rp[:, 0] * R + rp[:, 1] * leg], 1) / ds[:, None]
            dr = -np.stack([rp[:, 0] * leg + rp[:, 1] * R, -rp[:, 0] * R + rp[:, 1] * leg], 1) / ds[:, None]
            dirs = np.where(left[:, None], dl, dr)
            direction[legs] = dirs
            dot2 = (rel_vel[legs] * dirs).sum(1)
            u[legs] = dot2[:, None] * dirs - rel_vel[legs]

        # ---- already overlapping: resolve within one time step
        if colliding.any():
            inv_dt = 1.0 / dt
            wc = rel_vel[colliding] - inv_dt * rel_pos[colliding]
            wl = np.maximum(np.linalg.norm(wc, axis=1), 1e-9)
            unit_w = wc / wl[:, None]
            direction[colliding] = np.stack([unit_w[:, 1], -unit_w[:, 0]], 1)
            u[colliding] = (R * inv_dt - wl)[:, None] * unit_w

        point = vel[i] + 0.5 * u                      # reciprocal: half the effort each
        return _group(i, point, direction)

    def _obstacle_lines(self, state: SwarmState, ctx: StepContext, dt: float) -> dict[int, list]:
        det = ctx.detections
        if det is None or len(det.i) == 0:
            return {}
        r_eff = self.d.radius + self.cfg.safety_margin
        tau = self.cfg.orca_obstacle_horizon
        det = det.within(r_eff + self.d.max_speed * tau + 1.0)
        if len(det.i) == 0:
            return {}
        gap = det.dist - r_eff
        # allowed approach speed toward the point: gap / tau (or move out if inside)
        c = np.where(gap > 0, gap / tau, gap / dt)
        u = det.dir
        point = u * c[:, None]                          # constraint:  u . v <= c
        direction = np.stack([-u[:, 1], u[:, 0]], 1)
        return _group(det.i, point, direction)


def _group(i: np.ndarray, point: np.ndarray, direction: np.ndarray) -> dict[int, list]:
    out: dict[int, list] = {}
    for k, p, d in zip(i.tolist(), point.tolist(), direction.tolist()):
        out.setdefault(k, []).append((p[0], p[1], d[0], d[1]))
    return out


# ---------------------------------------------------------------------------
# 2D linear programs (port of RVO2's linearProgram1/2/3).
# A line is (px, py, dx, dy); the permitted half-plane is to the LEFT of the
# direction, i.e. det(d, p - v) <= 0.
# ---------------------------------------------------------------------------

def _det(ax, ay, bx, by):
    return ax * by - ay * bx


def _lp1(lines, no, radius, opt, direction_opt):
    px, py, dx, dy = lines[no]
    dot = px * dx + py * dy
    disc = dot * dot + radius * radius - (px * px + py * py)
    if disc < 0.0:
        return None                     # max-speed circle fully invalidates this line
    sq = math.sqrt(disc)
    t_left, t_right = -dot - sq, -dot + sq
    for k in range(no):
        qx, qy, ex, ey = lines[k]
        denom = _det(dx, dy, ex, ey)
        numer = _det(ex, ey, px - qx, py - qy)
        if abs(denom) <= EPS:
            if numer < 0.0:
                return None             # parallel and infeasible
            continue
        t = numer / denom
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None
    if direction_opt:
        t = t_right if opt[0] * dx + opt[1] * dy > 0.0 else t_left
    else:
        t = dx * (opt[0] - px) + dy * (opt[1] - py)
        t = min(max(t, t_left), t_right)
    return (px + t * dx, py + t * dy)


def _lp2(lines, radius, opt, direction_opt, _start):
    if direction_opt:
        result = (opt[0] * radius, opt[1] * radius)
    elif opt[0] * opt[0] + opt[1] * opt[1] > radius * radius:
        n = math.hypot(opt[0], opt[1])
        result = (opt[0] / n * radius, opt[1] / n * radius)
    else:
        result = opt
    for k, (px, py, dx, dy) in enumerate(lines):
        if _det(dx, dy, px - result[0], py - result[1]) > 0.0:
            new = _lp1(lines, k, radius, opt, direction_opt)
            if new is None:
                return k, result
            result = new
    return len(lines), result


def _lp3(lines, num_obst, begin, radius, result):
    distance = 0.0
    for k in range(begin, len(lines)):
        px, py, dx, dy = lines[k]
        if _det(dx, dy, px - result[0], py - result[1]) > distance:
            proj = list(lines[:num_obst])
            for m in range(num_obst, k):
                qx, qy, ex, ey = lines[m]
                determinant = _det(dx, dy, ex, ey)
                if abs(determinant) <= EPS:
                    if dx * ex + dy * ey > 0.0:
                        continue        # same direction: line m is redundant
                    lp = (0.5 * (px + qx), 0.5 * (py + qy))
                else:
                    t = _det(ex, ey, px - qx, py - qy) / determinant
                    lp = (px + t * dx, py + t * dy)
                ddx, ddy = ex - dx, ey - dy
                nl = math.hypot(ddx, ddy) or 1.0
                proj.append((lp[0], lp[1], ddx / nl, ddy / nl))
            temp = result
            fail, new = _lp2(proj, radius, (-dy, dx), True, result)
            result = temp if fail < len(proj) else new
            distance = _det(dx, dy, px - result[0], py - result[1])
    return result
