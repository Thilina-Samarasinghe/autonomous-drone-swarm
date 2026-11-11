"""Leader-follower formation flying.

Roles
  * the leader (lowest id, drone 0) flies toward the goal, using the
    path planner's waypoint if planning is on, or a patrol loop if
    there is no goal. It slows down while the formation is badly out of
    shape, so followers can catch up.
  * every follower holds a slot: a fixed offset in the leader's frame
    (forward = leader heading). It learns where the leader is from the
    radio beacon table (Phase 3), which is up to a few hundred ms old,
    and extrapolates with the leader's broadcast velocity.

Shape changes
  * slots are (re)assigned with the Hungarian algorithm, so a shape switch
    moves the drones the smallest total distance
  * the leader leaves a breadcrumb trail (its own flown path, shared over the
    radio). `column` places follower j on that trail j x `spacing` metres
    behind the leader, so the column follows the leader's exact route around
    corners instead of cutting through walls like a rigid line would

Squeezing through gaps (adaptive)
  * the leader measures how wide the passage is along its route for the next
    `squeeze_lookahead` metres: 2 x the smallest distance from the route to a
    known obstacle on its own map. If the shape would not fit, the formation
    switches to a column
  * once the passage ahead of the leader has been wide again for
    `restore_delay` seconds, the shape is restored. Followers still behind
    the narrow part then find a wall between them and their slot, so they
    follow the leader's trail through it (see below) and rejoin on the far side
  * any follower whose own map shows a wall between it and its slot follows
    the leader's trail instead: it flies to the newest trail point it can
    see, but always at least a few metres further along the trail than where
    it is, so it keeps moving even when a wall corner hides what's ahead
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from swarm.config import FormationConfig
from swarm.formation.shapes import slot_offsets, to_world, width

PATROL_MARGIN = (40.0, 30.0)        # patrol loop inset from the arena edges when there is no goal
TRAIL_STEP = 1.0                    # metres between breadcrumbs
TRAIL_MAX = 600                     # breadcrumbs kept
TRAIL_SEARCH = 160                  # newest breadcrumbs a blocked follower looks at
TRAIL_ADVANCE = 3                   # breadcrumbs (~m) a blocked follower always moves ahead


@dataclass
class FormationTargets:
    """What the formation controller needs this step."""
    leader: int
    leader_aim: np.ndarray           # (2,) where the leader is heading
    leader_speed: float              # leader's speed for this step
    slot: np.ndarray                 # (N, 2) target position per follower (nan = no target)
    ref_vel: np.ndarray              # (N, 2) feed-forward velocity per follower


@dataclass
class FormationStats:
    switches: int = 0                # slot (re)assignments
    squeezes: int = 0                # times the formation collapsed to a column for a gap
    error_sum: float = 0.0
    in_shape_steps: int = 0          # steps with mean slot error below IN_SHAPE_ERROR
    steps: int = 0
    last_error: float = 0.0

    @property
    def mean_error(self) -> float:
        return self.error_sum / max(self.steps, 1)

    @property
    def in_shape_fraction(self) -> float:
        return self.in_shape_steps / max(self.steps, 1)


IN_SHAPE_ERROR = 2.5                 # metres: "the formation is holding its shape"


class FormationManager:
    def __init__(self, n: int, cfg: FormationConfig, dt: float, world_size: np.ndarray):
        self.cfg = cfg
        self.dt = dt
        self.n = n
        self.leader = 0
        self.desired_shape = cfg.shape
        self.active_shape = cfg.shape
        self.offsets = slot_offsets(self.active_shape, n, cfg.spacing)
        self.assign = np.arange(n)                     # drone -> slot index
        self.heading: float | None = None
        self.need_assign = True
        self.squeezed = False
        self.clear_time = 0.0
        self.free_width = float("inf")
        mx, my = PATROL_MARGIN
        w, h = float(world_size[0]), float(world_size[1])
        self.patrol = np.array([[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]])
        self.patrol_idx = 0
        self.trail: list[np.ndarray] = []              # leader breadcrumbs, oldest first
        self.following_trail = np.zeros(n, dtype=bool)
        self.stats = FormationStats()

    # ------------------------------------------------------------- control
    def set_shape(self, shape: str) -> None:
        self.desired_shape = shape
        if not self.squeezed:
            self._activate(shape)

    def _activate(self, shape: str) -> None:
        self.active_shape = shape
        self.offsets = slot_offsets(shape, self.n, self.cfg.spacing)
        self.need_assign = True

    def _assign(self, pos: np.ndarray, origin: np.ndarray) -> None:
        """Hungarian assignment of followers to slots (minimum total distance)."""
        followers = np.array([k for k in range(self.n) if k != self.leader])
        self.assign = np.zeros(self.n, dtype=int)
        self.assign[self.leader] = 0
        if len(followers):
            slots = to_world(self.offsets[1:], origin, self.heading)
            cost = np.linalg.norm(pos[followers][:, None] - slots[None], axis=2)
            rows, cols = linear_sum_assignment(cost)
            self.assign[followers[rows]] = cols + 1
        self.need_assign = False
        self.stats.switches += 1

    # --------------------------------------------------------------- sensing
    def _route_ahead(self, pos_l: np.ndarray, route: np.ndarray | None) -> np.ndarray:
        """Points every ~1 m along the leader's next `squeeze_lookahead` metres:
        its planned path if it has one, otherwise straight along its heading."""
        look = self.cfg.squeeze_lookahead
        if route is not None and len(route) > 1:
            pts = np.vstack([pos_l[None], route])
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(seg)]
            arc = np.arange(2.0, look, 1.0)
            arc = arc[arc <= cum[-1]]
            if len(arc):
                i = np.clip(np.searchsorted(cum, arc, side="right") - 1, 0, len(seg) - 1)
                f = (arc - cum[i]) / np.maximum(seg[i], 1e-9)
                return pts[i] + (pts[i + 1] - pts[i]) * f[:, None]
        h = np.array([math.cos(self.heading), math.sin(self.heading)])
        return pos_l[None] + np.arange(2.0, look, 1.0)[:, None] * h[None]

    def _passage_width(self, pos_l: np.ndarray, route, maps) -> float:
        """2 x the smallest distance from the leader's route ahead to an obstacle
        it knows about (its own map). inf if it knows of nothing nearby."""
        if maps is None:
            return float("inf")
        occ = maps.grid(maps.L[self.leader]) >= maps.cfg.known_threshold
        if not occ.any():
            return float("inf")
        dist = ndimage.distance_transform_edt(~occ) * maps.cell          # metres to nearest obstacle
        pts = self._route_ahead(pos_l, route)
        cx = np.clip((pts[:, 0] // maps.cell).astype(int), 0, maps.gw - 1)
        cy = np.clip((pts[:, 1] // maps.cell).astype(int), 0, maps.gh - 1)
        # distance is measured from cell centres to obstacle cell centres: add one cell
        return float(2.0 * (dist[cy, cx].min() + 0.5 * maps.cell))

    def _update_squeeze(self, pos, route, maps) -> None:
        if not self.cfg.adaptive:
            return
        L = self.leader
        self.free_width = self._passage_width(pos[L], route, maps)
        needed = width(slot_offsets(self.desired_shape, self.n, self.cfg.spacing)) + 2 * self.cfg.squeeze_margin
        if not self.squeezed:
            if self.free_width < needed and self.desired_shape != "column":
                self.squeezed = True
                self.clear_time = 0.0
                self.stats.squeezes += 1
                self._activate("column")
            return
        # squeezed: wait until the gap is behind the leader ...
        if self.free_width >= needed + 2.0:
            self.clear_time += self.dt
        else:
            self.clear_time = 0.0
            return
        if self.clear_time >= self.cfg.restore_delay:
            self.squeezed = False
            self._activate(self.desired_shape)

    # ---------------------------------------------------------------- update
    def update(self, time: float, pos: np.ndarray, vel: np.ndarray, scan, goal: np.ndarray | None,
               planner_targets: np.ndarray | None, comms, perception_radius: float,
               maps=None, planner_paths=None, planner_progress=None) -> FormationTargets:
        n, L, cfg = self.n, self.leader, self.cfg
        # leader heading, low-pass filtered so the formation doesn't jitter
        speed = float(np.linalg.norm(vel[L]))
        target_h = math.atan2(vel[L, 1], vel[L, 0]) if speed > 0.5 else self.heading
        if self.heading is None:
            self.heading = target_h if target_h is not None else 0.0
        elif target_h is not None:
            d = (target_h - self.heading + math.pi) % (2 * math.pi) - math.pi
            self.heading += d * min(1.0, self.dt / cfg.heading_tau)

        # what the leader believes about its followers (its own radio table)
        if comms is not None:
            beliefs = comms.table[L, :, :2].copy()
        else:
            beliefs = pos.copy()
        beliefs[L] = pos[L]

        route = None
        if planner_paths is not None and planner_paths[L] is not None:
            route = planner_paths[L][planner_progress[L]:]
        self._update_squeeze(pos, route, maps)
        if self.need_assign:
            self._assign(pos, pos[L])

        # leader aim: planner waypoint > goal > patrol loop
        if goal is not None:
            aim = planner_targets[L] if planner_targets is not None else goal
        else:
            if np.linalg.norm(self.patrol[self.patrol_idx] - pos[L]) < 8.0:
                self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol)
            aim = self.patrol[self.patrol_idx]

        # each follower's estimate of the leader (pos + vel) from the radio, extrapolated
        slot = np.full((n, 2), np.nan)
        ref = np.zeros((n, 2))
        est_pos, est_vel, fresh = self._estimates(time, pos, vel, comms, perception_radius, L)
        R = np.array([[math.cos(self.heading), -math.sin(self.heading)],
                      [math.sin(self.heading), math.cos(self.heading)]])
        followers = np.array([k for k in range(n) if k != L], dtype=int)

        self._record_trail(pos[L])
        self.following_trail[:] = False
        ok = fresh[followers]
        f = followers[ok]
        if self.active_shape == "column":
            # follower in slot j sits j * spacing metres back along the leader's trail
            pts, tang = self._trail_points(self.assign[f] * cfg.spacing)
            slot[f] = pts
            ref[f] = tang * float(np.linalg.norm(vel[L]))
        else:
            slot[f] = est_pos[f] + self.offsets[self.assign[f]] @ R.T
            ref[f] = est_vel[f]
            # a follower whose own map shows a wall between it and its slot
            # can't fly there directly: it follows the leader's trail instead
            if maps is not None and len(f) and len(self.trail) > 1:
                blocked = f[~maps.segments_clear(f, pos[f], slot[f])]
                if len(blocked):
                    slot[blocked] = self._trail_waypoints(blocked, pos, maps)
                    ref[blocked] = 0.0
                    self.following_trail[blocked] = True

        # formation error (true positions) and the leader's slow-down from ITS beliefs
        true_err = self._true_error(pos, R)
        self.stats.steps += 1
        self.stats.error_sum += true_err
        self.stats.in_shape_steps += int(true_err < IN_SHAPE_ERROR)
        self.stats.last_error = true_err
        believed = self._believed_error(beliefs, R, maps)
        scale = float(np.clip(1.0 - (believed - cfg.slowdown_error) / (2 * cfg.slowdown_error), 0.35, 1.0))
        return FormationTargets(L, np.asarray(aim, dtype=float), cfg.speed * scale, slot, ref)

    # ----------------------------------------------------------------- trail
    def _record_trail(self, p: np.ndarray) -> None:
        if not self.trail or np.linalg.norm(p - self.trail[-1]) >= TRAIL_STEP:
            self.trail.append(p.copy())
            if len(self.trail) > TRAIL_MAX:
                del self.trail[0]

    def _trail_points(self, arc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Points `arc` metres back along the trail from the leader, and the
        unit direction of travel there. Beyond the trail's start, extend it
        straight back from the oldest breadcrumb."""
        back = np.array(self.trail[::-1])                       # newest (leader) first
        heading = np.array([math.cos(self.heading), math.sin(self.heading)])
        if len(back) < 2:
            return back[0] - arc[:, None] * heading, np.repeat(heading[None], len(arc), 0)
        seg = np.diff(back, axis=0)                             # pointing backwards in time
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.r_[0.0, np.cumsum(seg_len)]
        i = np.clip(np.searchsorted(cum, arc, side="right") - 1, 0, len(seg) - 1)
        frac = np.clip((arc - cum[i]) / np.maximum(seg_len[i], 1e-9), 0.0, None)
        pts = back[i] + seg[i] * frac[:, None]                  # frac > 1 only past the end: extends
        tang = -seg[i] / np.maximum(seg_len[i], 1e-9)[:, None]  # direction of travel
        return pts, tang

    def _trail_waypoints(self, ks: np.ndarray, pos: np.ndarray, maps) -> np.ndarray:
        """For each blocked follower: the newest breadcrumb it can see in a
        straight line on its own map, but never less than TRAIL_ADVANCE
        breadcrumbs beyond the one it is nearest to (keeps it moving along
        the trail when a wall corner hides what lies ahead)."""
        crumbs = np.array(self.trail[-TRAIL_SEARCH:])           # oldest first
        m, t = len(ks), len(crumbs)
        kk = np.repeat(ks, t)
        p = np.repeat(pos[ks], t, axis=0)
        q = np.tile(crumbs, (m, 1))
        visible = maps.segments_clear(kk, p, q).reshape(m, t)
        newest_visible = np.where(visible.any(axis=1), t - 1 - visible[:, ::-1].argmax(axis=1), -1)
        nearest = np.linalg.norm(crumbs[None] - pos[ks][:, None], axis=2).argmin(axis=1)
        pick = np.maximum(newest_visible, np.minimum(nearest + TRAIL_ADVANCE, t - 1))
        return crumbs[pick]

    # --------------------------------------------------------------- helpers
    def _estimate_of(self, k, j, time, pos, vel, comms, perception_radius):
        """Drone k's estimate of drone j: sensed directly if close, else from the radio."""
        if np.linalg.norm(pos[j] - pos[k]) <= perception_radius or comms is None:
            return pos[j], vel[j], True
        p = comms.table[k, j]
        if np.isnan(p[2]) or time - p[2] > self.cfg.leader_timeout:
            return None, None, False
        v = comms.vtable[k, j]
        return p[:2] + v * (time - p[2]), v, True

    def _estimates(self, time, pos, vel, comms, perception_radius, L):
        n = self.n
        est_pos = np.repeat(pos[L][None], n, axis=0)
        est_vel = np.repeat(vel[L][None], n, axis=0)
        fresh = np.ones(n, dtype=bool)
        if comms is None:
            return est_pos, est_vel, fresh
        seen = np.linalg.norm(pos - pos[L], axis=1) <= perception_radius
        p = comms.table[:, L]
        age = time - p[:, 2]
        radio = ~seen
        fresh = seen | (~np.isnan(p[:, 2]) & (age <= self.cfg.leader_timeout))
        use = radio & fresh
        est_vel[use] = comms.vtable[use, L]
        est_pos[use] = p[use, :2] + est_vel[use] * age[use, None]
        return est_pos, est_vel, fresh

    def _ideal_slots(self, origin, R):
        return origin[None] + self.offsets[self.assign] @ R.T

    def follower_errors(self, pos: np.ndarray) -> np.ndarray:
        """Distance of every follower from where it should be (TRUE positions):
        its formation slot, or its point on the leader's trail in a column. (N-1,)"""
        n, L = self.n, self.leader
        mask = np.arange(n) != L
        if n < 2 or self.heading is None:
            return np.zeros(int(mask.sum()))
        if self.active_shape == "column" and len(self.trail):
            pts, _ = self._trail_points(self.assign[mask] * self.cfg.spacing)
            return np.linalg.norm(pos[mask] - pts, axis=1)
        R = np.array([[math.cos(self.heading), -math.sin(self.heading)],
                      [math.sin(self.heading), math.cos(self.heading)]])
        return np.linalg.norm(pos[mask] - self._ideal_slots(pos[L], R)[mask], axis=1)

    def _true_error(self, pos, R) -> float:
        e = self.follower_errors(pos)
        return float(e.mean()) if len(e) else 0.0

    def _believed_error(self, beliefs, R, maps) -> float:
        """Mean slot error the LEADER believes in, used to slow it down.
        Followers it believes are behind a wall from their slot are expected
        to be off-slot (they're following the trail) and don't count."""
        L = self.leader
        known = ~np.isnan(beliefs[:, 0])
        known[L] = False
        if not known.any() or self.active_shape == "column":
            return 0.0                     # a column follows the trail; don't stall the leader
        ideal = self._ideal_slots(beliefs[L], R)
        idx = np.flatnonzero(known)
        if maps is not None:
            idx = idx[maps.segments_clear(np.full(len(idx), L), beliefs[idx], ideal[idx])]
        if len(idx) == 0:
            return 0.0
        return float(np.linalg.norm(beliefs[idx] - ideal[idx], axis=1).mean())
