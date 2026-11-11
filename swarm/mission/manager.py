"""Cooperative search-and-rescue mission.

Every drone runs the same state machine on its OWN knowledge:

    EXPLORE --(hears a victim and wins the auction)--> SEEK
    SEEK    --(victim confirmed / lost auction / timeout)--> EXPLORE
    any     --(battery only just enough to get home)--> RETURN --> CHARGE --> EXPLORE
    any     --(battery empty)--> DEAD

Exploration (where to look next)
  * frontier : fly to the nearest frontier cell of my own map
               (known-free cell next to unknown space)
  * voronoi  : same, but only frontiers that are closer to me than to any
               teammate I know about (radio beacon table): the area is split
               between drones dynamically, so they spread out instead of all
               chasing the same frontier
  * random   : fly to random points (baseline)

Seeking (finding a victim I can hear) - Particle Swarm Optimisation
  Each seeker is a particle. Every reading (4 Hz) it updates
      v <- w v + c1 r1 (p_best - x) + c2 r2 (g_best - x) + jitter
  p_best = where IT measured the strongest signal of that victim,
  g_best = the strongest reading any teammate shared over the radio.
  Two additions make it work with a handful of particles and noisy, weak
  signals:
    * the jitter scales with the distance the best reading implies (the
      signal model gives a rough range), so far-off particles explore widely
      and close-in ones fine-tune
    * if g_best hasn't improved for STALL_TIME seconds, each seeker gets a
      random kick to sample new ground (a standard PSO "restart")
  Only drones that hear the victim, or are within signal range of its best
  known reading, may join the search.

Task allocation - a consensus auction (simplified CBBA, one task per drone)
  A drone that wants a victim bids its distance to the best-known signal
  location. Bids spread over the radio; a drone keeps chasing only while it
  is among the `max_seekers` lowest bidders it knows of.

Knowledge shared over the radio (piggy-backing on Phase 3 deliveries)
  * which victims are confirmed (and where)
  * best signal reading per victim (value + position)  -> PSO g_best
  * auction bids
The base station hears every drone within radio range; a victim counts as
*reported* once the base knows it was confirmed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from swarm.config import MissionConfig
from swarm.mission.victims import REFERENCE_DISTANCE, VictimField

EXPLORE, SEEK, RETURN, CHARGE, DEAD = range(5)
MODE_NAMES = ("explore", "seek", "return", "charge", "dead")
BID_FRESH = 5.0          # s: bids older than this are ignored
GBEST_FRESH = 30.0       # s: a shared best reading older than this is stale
IGNORE_AFTER_TIMEOUT = 30.0
STALL_TIME = 6.0         # s without a better g_best -> PSO restart kick
BLACKLIST = 30           # remembered unreachable exploration targets per drone
STUCK_TIME = 8.0         # s spent within STUCK_RADIUS while trying to go somewhere = stuck
STUCK_RADIUS = 2.0       # m
ESCAPE_TIME = 3.0        # s of escape manoeuvre for a stuck returning drone


@dataclass
class MissionStats:
    found_time: np.ndarray            # (V,) first camera confirmation (nan = not yet)
    reported_time: np.ndarray         # (V,) base station learned it (nan = not yet)
    returns: int = 0                  # trips home to recharge
    stuck: int = 0                    # times a drone stopped making progress and recovered
    lost: int = 0                     # drones whose battery ran out
    distance: float = 0.0             # metres flown by the whole swarm
    energy: float = 0.0               # battery units used
    mode_steps: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def found(self) -> int:
        return int((~np.isnan(self.found_time)).sum())

    @property
    def reported(self) -> int:
        return int((~np.isnan(self.reported_time)).sum())

    @property
    def complete_time(self) -> float:
        return float(np.nanmax(self.reported_time)) if self.reported == len(self.reported_time) else float("nan")


class RescueMission:
    def __init__(self, n: int, victims: np.ndarray, base: np.ndarray, cfg: MissionConfig,
                 world_size: np.ndarray, dt: float, comm_range: float, rng: np.random.Generator,
                 edge_margin: float | None = None):
        self.n, self.cfg, self.dt = n, cfg, dt
        self.rng = rng
        self.base = np.asarray(base, dtype=float)
        self.world_size = np.asarray(world_size, dtype=float)
        self.comm_range = comm_range
        # frontiers inside the arena's soft-wall zone are ignored (lidar sees there anyway)
        self.edge_margin = max(cfg.edge_margin, edge_margin or 0.0)
        self.anchor = np.zeros((n, 2))                 # stuck detection: where / when the
        self.anchor_time = np.zeros(n)                 # drone last moved STUCK_RADIUS metres
        self.escape_until = np.full(n, -np.inf)
        self.escape_dir = np.zeros((n, 2))
        self.field = VictimField(victims, cfg.signal_range, cfg.signal_noise, rng)
        V = self.field.count
        self.V = V
        self.mode = np.full(n, EXPLORE)
        self.battery = np.full(n, cfg.battery_capacity)
        self.target = np.full((n, 2), np.nan)          # exploration / return target
        self.next_retarget = np.zeros(n, dtype=int)
        self.blacklist: list[list[np.ndarray]] = [[] for _ in range(n)]
        self.seek_v = np.full(n, -1)
        self.seek_since = np.zeros(n)
        self.ignore_until = np.zeros((n, V))
        self.pso_v = np.zeros((n, 2))
        self.gbest_improved = np.zeros((n, V))         # when drone k's g_best for v last improved
        self.pbest_val = np.zeros((n, V))
        self.pbest_pos = np.zeros((n, V, 2))
        # knowledge (row = what drone k knows)
        self.found = np.zeros((n, V), dtype=bool)
        self.gbest_val = np.zeros((n, V))
        self.gbest_pos = np.zeros((n, V, 2))
        self.gbest_time = np.full((n, V), -np.inf)
        self.bid = np.full((n, V, n), np.inf)          # what k believes j bid for v
        self.bid_time = np.full((n, V, n), -np.inf)
        self.base_known = np.zeros(V, dtype=bool)
        self.last_reading = np.zeros((n, V))
        self.read_every = max(1, round(0.25 / dt))
        self.retarget_steps = max(1, round(cfg.retarget_interval / dt))
        self.stats = MissionStats(np.full(V, np.nan), np.full(V, np.nan))

    # ---------------------------------------------------------------- helpers
    def _drain(self, speed: float) -> float:
        return self.cfg.drain_hover + self.cfg.drain_speed * speed ** 2

    def _needed_to_return(self, pos: np.ndarray) -> np.ndarray:
        """Battery needed to fly home (1.5x straight-line distance) plus the reserve."""
        d = np.linalg.norm(pos - self.base, axis=1) * 1.5
        v = self.cfg.explore_speed
        return self.cfg.reserve + d / v * self._drain(v)

    # --------------------------------------------------------------- update
    def update(self, step: int, time: float, pos: np.ndarray, vel: np.ndarray,
               maps, comms) -> None:
        """Advance every drone's mission state by one step (before control)."""
        cfg, dt = self.cfg, self.dt
        alive = self.mode != DEAD
        speed = np.linalg.norm(vel, axis=1)

        # ---- battery
        use = np.where(alive, (cfg.drain_hover + cfg.drain_speed * speed ** 2) * dt, 0.0)
        charging = self.mode == CHARGE
        self.battery -= np.where(charging, 0.0, use)
        self.battery[charging] = np.minimum(self.battery[charging] + cfg.charge_rate * dt,
                                            cfg.battery_capacity)
        self.stats.energy += float(use[~charging].sum())
        self.stats.distance += float((speed * dt)[alive].sum())
        dead_now = alive & (self.battery <= 0.0)
        if dead_now.any():
            self.mode[dead_now] = DEAD
            self.battery[dead_now] = 0.0
            self.stats.lost += int(dead_now.sum())
            alive &= ~dead_now

        # ---- camera: confirm any victim within detect radius
        d_true = self.field.distances(pos)
        seen = (d_true < cfg.detect_radius) & alive[:, None]
        if seen.any():
            new = seen & ~self.found
            self.found |= seen
            first = new.any(axis=0) & np.isnan(self.stats.found_time)
            self.stats.found_time[first] = time

        # ---- radio: signal readings, gossip, base station
        if step % self.read_every == 0:
            self._read_signals(time, pos, alive)
        if comms is not None and len(comms.deliveries):
            self._gossip(comms.deliveries, alive)
        if comms is None:
            # no radio: only drones at the base can report
            at_base = np.linalg.norm(pos - self.base, axis=1) < cfg.charge_radius * 1.5
            self._report(at_base & alive, time)
        elif step % comms.interval_steps == 0:
            in_range = np.linalg.norm(pos - self.base, axis=1) <= self.comm_range
            ok = self.rng.random(self.n) >= comms.cfg.packet_loss
            self._report(in_range & ok & alive, time)

        # ---- state machine
        need = self._needed_to_return(pos)
        at_base = np.linalg.norm(pos - self.base, axis=1) < cfg.charge_radius
        for k in np.flatnonzero(alive):
            m = self.mode[k]
            if m in (EXPLORE, SEEK) and self.battery[k] <= need[k]:
                self._leave_seek(k)
                self.mode[k] = RETURN
                self.target[k] = self.base
                self.stats.returns += 1
            elif m == RETURN and at_base[k]:
                self.mode[k] = CHARGE
            elif m == CHARGE and self.battery[k] >= cfg.battery_capacity * 0.98:
                self.mode[k] = EXPLORE
                self.next_retarget[k] = step
            elif m == SEEK:
                v = self.seek_v[k]
                if (self.found[k, v] or not self._wins(k, v, time)
                        or time - self.seek_since[k] > cfg.seek_timeout):
                    if time - self.seek_since[k] > cfg.seek_timeout:
                        self.ignore_until[k, v] = time + IGNORE_AFTER_TIMEOUT
                    self._leave_seek(k)
                    self.mode[k] = EXPLORE
                    self.next_retarget[k] = step
        # explorers: maybe start seeking, else keep a good exploration target
        explorers = np.flatnonzero(self.mode == EXPLORE)
        if cfg.seek == "pso":
            for k in explorers:
                v = self._choose_victim(k, pos[k], time)
                if v >= 0:
                    self._start_seek(k, v, pos[k], vel[k], time)
        explorers = np.flatnonzero(self.mode == EXPLORE)
        due = [k for k in explorers if step >= self.next_retarget[k]
               or np.isnan(self.target[k, 0])
               or np.linalg.norm(pos[k] - self.target[k]) < 4.0]
        if due:
            self._retarget(np.array(due), pos, maps, comms, time)
            self.next_retarget[due] = step + self.retarget_steps
        self._check_progress(time, pos)
        for mo in range(5):
            self.stats.mode_steps[mo] += int((self.mode == mo).sum())

    def _check_progress(self, time, pos) -> None:
        """Stuck detection (safety net): a navigating drone that has stayed
        within STUCK_RADIUS of one spot for STUCK_TIME seconds."""
        nav = ((self.mode == EXPLORE) | (self.mode == RETURN)) & ~np.isnan(self.target[:, 0])
        moved = np.linalg.norm(pos - self.anchor, axis=1) > STUCK_RADIUS
        reset = moved | ~nav
        self.anchor[reset] = pos[reset]
        self.anchor_time[reset] = time
        stuck = nav & (time - self.anchor_time > STUCK_TIME) & (time >= self.escape_until)
        for k in np.flatnonzero(stuck):
            self.stats.stuck += 1
            if self.mode[k] == EXPLORE:
                self.unreachable(k)                     # give up on this target, pick another
            else:                                       # returning: back off toward open space
                centre = self.world_size / 2 - pos[k]
                n = np.linalg.norm(centre)
                jitter = self.rng.normal(0.0, 0.5, 2)
                self.escape_dir[k] = centre / max(n, 1e-6) + jitter
                self.escape_until[k] = time + ESCAPE_TIME
            self.anchor_time[k] = time

    # ----------------------------------------------------------- reporting
    def _report(self, senders: np.ndarray, time: float) -> None:
        if not senders.any():
            return
        known = self.found[senders].any(axis=0)
        new = known & ~self.base_known
        self.base_known |= known
        self.stats.reported_time[new & np.isnan(self.stats.reported_time)] = time

    # ------------------------------------------------------------ signals
    def _read_signals(self, time, pos, alive) -> None:
        r = self.field.read(pos)
        r[~alive] = 0.0
        self.last_reading = r
        better = r > self.pbest_val
        self.pbest_val = np.where(better, r, self.pbest_val)
        self.pbest_pos = np.where(better[..., None], pos[:, None, :], self.pbest_pos)
        own = r > self.gbest_val
        self.gbest_improved = np.where(own, time, self.gbest_improved)
        self.gbest_val = np.where(own, r, self.gbest_val)
        self.gbest_pos = np.where(own[..., None], pos[:, None, :], self.gbest_pos)
        self.gbest_time = np.where(r > 0, time, self.gbest_time)

    def _gossip(self, pairs: np.ndarray, alive: np.ndarray) -> None:
        pairs = pairs[alive[pairs[:, 0]] & alive[pairs[:, 1]]]
        if len(pairs) == 0:
            return
        s, r = pairs[:, 0], pairs[:, 1]
        # confirmed victims: logical OR
        acc = np.zeros((self.n, self.V))
        np.add.at(acc, r, self.found[s])
        self.found |= acc > 0
        # best reading per victim: keep the maximum (and its position)
        old = self.gbest_val.copy()
        new = old.copy()
        np.maximum.at(new, r, self.gbest_val[s])
        take = (self.gbest_val[s] >= new[r]) & (self.gbest_val[s] > old[r])      # (P, V)
        pi, vi = np.nonzero(take)
        self.gbest_val[r[pi], vi] = self.gbest_val[s[pi], vi]
        self.gbest_pos[r[pi], vi] = self.gbest_pos[s[pi], vi]
        self.gbest_time[r[pi], vi] = np.maximum(self.gbest_time[r[pi], vi], self.gbest_time[s[pi], vi])
        self.gbest_improved[r[pi], vi] = np.maximum(self.gbest_improved[r[pi], vi],
                                                    self.gbest_improved[s[pi], vi])
        # auction bids: keep the newer entry
        newer = self.bid_time[s] > self.bid_time[r]                               # (P, V, N)
        pi, vi, ji = np.nonzero(newer)
        self.bid[r[pi], vi, ji] = self.bid[s[pi], vi, ji]
        self.bid_time[r[pi], vi, ji] = self.bid_time[s[pi], vi, ji]

    # ------------------------------------------------------------- auction
    def _wins(self, k: int, v: int, time: float) -> bool:
        """Am I among the `max_seekers` lowest fresh bidders I know of for v?"""
        fresh = self.bid_time[k, v] >= time - BID_FRESH
        bids = np.where(fresh, self.bid[k, v], np.inf)
        mine = bids[k]
        if not np.isfinite(mine):
            return False
        better = (bids < mine) | ((bids == mine) & (np.arange(self.n) < k))
        return int(better.sum()) < self.cfg.max_seekers

    def _place_bid(self, k: int, v: int, pos_k: np.ndarray, time: float) -> None:
        target = self.gbest_pos[k, v] if self.gbest_val[k, v] > 0 else pos_k
        self.bid[k, v, k] = float(np.linalg.norm(pos_k - target))
        self.bid_time[k, v, k] = time

    def _choose_victim(self, k: int, pos_k: np.ndarray, time: float) -> int:
        """The strongest unconfirmed victim I hear (or know a fresh reading of)
        for which I win the auction; -1 if none."""
        heard = self.last_reading[k] > 0
        near = np.linalg.norm(self.gbest_pos[k] - pos_k, axis=1) <= self.cfg.signal_range
        known = (self.gbest_val[k] > 0) & (self.gbest_time[k] >= time - GBEST_FRESH) & near
        cand = (heard | known) & ~self.found[k] & (self.ignore_until[k] <= time)
        if not cand.any():
            return -1
        order = np.argsort(-np.where(cand, np.maximum(self.last_reading[k], self.gbest_val[k] * 0.5), -1))
        for v in order[:int(cand.sum())]:
            self._place_bid(k, int(v), pos_k, time)
            if self._wins(k, int(v), time):
                return int(v)
            self.bid[k, v, k] = np.inf            # withdraw: someone closer has it
        return -1

    def _start_seek(self, k, v, pos_k, vel_k, time) -> None:
        self.mode[k] = SEEK
        self.seek_v[k] = v
        self.seek_since[k] = time
        g = self.gbest_pos[k, v] - pos_k
        n = np.linalg.norm(g)
        self.pso_v[k] = g / n * self.cfg.pso_max_speed if n > 1e-6 else vel_k

    def _leave_seek(self, k) -> None:
        v = self.seek_v[k]
        if v >= 0:
            self.bid[k, v, k] = np.inf
        self.seek_v[k] = -1

    # ---------------------------------------------------------- exploration
    def _frontiers(self, maps, k: int) -> np.ndarray:
        """World positions of frontier cells on drone k's map (known free next to unknown)."""
        thr = maps.cfg.known_threshold
        g = maps.grid(maps.L[k])
        free = g <= -thr
        unknown = np.abs(g) < thr
        nb = np.zeros_like(unknown)
        nb[1:, :] |= unknown[:-1, :]
        nb[:-1, :] |= unknown[1:, :]
        nb[:, 1:] |= unknown[:, :-1]
        nb[:, :-1] |= unknown[:, 1:]
        ys, xs = np.nonzero(free & nb)
        pts = np.stack([(xs + 0.5) * maps.cell, (ys + 0.5) * maps.cell], axis=1)
        m = self.edge_margin
        inside = ((pts[:, 0] > m) & (pts[:, 0] < self.world_size[0] - m)
                  & (pts[:, 1] > m) & (pts[:, 1] < self.world_size[1] - m))
        return pts[inside]

    def _retarget(self, ks: np.ndarray, pos, maps, comms, time) -> None:
        cfg = self.cfg
        for k in ks:
            if cfg.explore == "random" or maps is None:
                self.target[k] = self.rng.uniform([cfg.edge_margin * 2] * 2,
                                                  self.world_size - cfg.edge_margin * 2)
                continue
            F = self._frontiers(maps, k)
            for b in self.blacklist[k]:
                if len(F):
                    F = F[np.linalg.norm(F - b, axis=1) > 4.0]
            if len(F) == 0:
                self.target[k] = self.rng.uniform([cfg.edge_margin * 2] * 2,
                                                  self.world_size - cfg.edge_margin * 2)
                continue
            d_me = np.linalg.norm(F - pos[k], axis=1)
            if cfg.explore == "voronoi":
                mates = self._teammates(k, pos, comms, time)
                if len(mates):
                    d_other = np.linalg.norm(F[:, None] - mates[None], axis=2).min(axis=1)
                    mine = d_me <= d_other
                    if mine.any():
                        F, d_me = F[mine], d_me[mine]
            self.target[k] = F[int(d_me.argmin())]

    def _teammates(self, k: int, pos, comms, time) -> np.ndarray:
        """Where drone k believes its (alive) teammates are."""
        others = np.ones(self.n, dtype=bool)
        others[k] = False
        others &= self.mode != DEAD
        others &= self.mode != CHARGE                  # charging drones aren't exploring
        if comms is None:
            near = np.linalg.norm(pos - pos[k], axis=1) < 15.0       # only what it can see
            return pos[others & near]
        t = comms.table[k]
        fresh = ~np.isnan(t[:, 2]) & (t[:, 2] >= time - 10.0)
        return t[others & fresh, :2]

    def unreachable(self, k: int) -> None:
        """Called when the planner can't find a route to drone k's target."""
        if not np.isnan(self.target[k, 0]):
            self.blacklist[k].append(self.target[k].copy())
            del self.blacklist[k][:-BLACKLIST]
            self.target[k] = np.nan

    # ------------------------------------------------------------- outputs
    def nav_goals(self) -> np.ndarray:
        """(N, 2) where each drone navigates with the planner (NaN = not navigating)."""
        goals = np.full((self.n, 2), np.nan)
        nav = (self.mode == EXPLORE) | (self.mode == RETURN)
        goals[nav] = self.target[nav]
        return goals

    def desired_velocity(self, step, pos, waypoints: np.ndarray | None,
                         time: float = 0.0) -> np.ndarray:
        """(N, 2) velocity each drone wants this step."""
        cfg = self.cfg
        out = np.zeros((self.n, 2))
        nav = ((self.mode == EXPLORE) | (self.mode == RETURN)) & ~np.isnan(self.target[:, 0])
        aim = waypoints if waypoints is not None else self.target
        aim = np.where(np.isnan(aim), self.target, aim)
        to = aim[nav] - pos[nav]
        dist_goal = np.linalg.norm(self.target[nav] - pos[nav], axis=1)
        speed = cfg.explore_speed * np.clip(dist_goal / 6.0, 0.3, 1.0)
        n = np.linalg.norm(to, axis=1, keepdims=True)
        out[nav] = np.where(n > 1e-6, to / np.maximum(n, 1e-6), 0.0) * speed[:, None]

        seek = np.flatnonzero(self.mode == SEEK)
        if len(seek) and step % self.read_every == 0:
            v = self.seek_v[seek]
            r1 = self.rng.random((len(seek), 1))
            r2 = self.rng.random((len(seek), 1))
            pb = self.pbest_pos[seek, v]
            gb = self.gbest_pos[seek, v]
            x = pos[seek]
            # range implied by the best reading (inverse of the signal model), metres
            best = np.clip(self.gbest_val[seek, v], 1e-3, 1.0)
            est_range = REFERENCE_DISTANCE * np.sqrt(np.maximum(1.0 / best - 1.0, 0.0))
            jitter = cfg.pso_jitter * np.clip(est_range / REFERENCE_DISTANCE, 0.3, 4.0)
            new = (cfg.pso_inertia * self.pso_v[seek]
                   + cfg.pso_cognitive * r1 * (pb - x)
                   + cfg.pso_social * r2 * (gb - x)
                   + self.rng.normal(0.0, 1.0, (len(seek), 2)) * jitter[:, None])
            stalled = time - self.gbest_improved[seek, v] > STALL_TIME
            if stalled.any():
                ang = self.rng.uniform(0, 2 * np.pi, int(stalled.sum()))
                new[stalled] = np.stack([np.cos(ang), np.sin(ang)], 1) * cfg.pso_max_speed
                self.gbest_improved[seek[stalled], v[stalled]] = time   # one kick per stall
            n = np.linalg.norm(new, axis=1, keepdims=True)
            self.pso_v[seek] = new * np.minimum(1.0, cfg.pso_max_speed / np.maximum(n, 1e-9))
        out[seek] = self.pso_v[seek]
        escaping = (self.mode == RETURN) & (time < self.escape_until)
        if escaping.any():
            d = self.escape_dir[escaping]
            out[escaping] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6) * cfg.explore_speed
        return out                                   # CHARGE / DEAD: hold still
