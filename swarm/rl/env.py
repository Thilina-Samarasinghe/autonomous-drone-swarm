"""Multi-agent training environments built on the real simulator.

Every drone is an agent; all agents share ONE policy (parameter sharing),
each acting on its own local observation (swarm/rl/observe.py). Actions are
desired velocities in [-1, 1]^2 x max_speed (see observe.action_to_accel). The simulator's physics,
lidar, walls and obstacles are exactly the ones used everywhere else.

Tasks and rewards (per drone, per step)
  navigate   reach your goal (a new one appears on arrival) among obstacles
             and other drones, with NO hand-written avoidance:
             + progress toward the goal, + bonus on arrival.
             With `shared_goal_prob`, some episodes give every drone the SAME
             goal (the crowded case the scenarios use).
  flock      fly as a flock, no goal:
             + alignment with neighbours, + having neighbours (up to 4),
             + speed near cruise, - being alone, - being too close
  formation  hold your slot in a V behind a scripted leader (drone 0):
             + getting closer to the slot, - distance from it,
             + bonus when within 1.5 m
  search     explore unknown space (own lidar map):
             + newly mapped cells
  all tasks  - collisions with drones or obstacles, - flying into walls,
             - near misses (smooth), - large accelerations
"""

from __future__ import annotations

import math

import numpy as np

from swarm.config import load_config
from swarm.formation.shapes import slot_offsets
from swarm.rl.observe import OBS_DIM, Observer, action_to_accel
from swarm.sim.engine import Simulation

TASKS = ("navigate", "flock", "formation", "search")

TASK_SETUP = {
    #             scenario   mapping  episode steps
    "navigate": ("random", False, 400),
    "flock": ("open", False, 400),
    "formation": ("open", False, 400),
    "search": ("survey", True, 600),
}


class _Hold:
    """Controller that returns whatever acceleration the environment set."""
    def __init__(self):
        self.accel = None

    def compute(self, state, ctx):
        return self.accel if self.accel is not None else np.zeros((state.n, 2))


class SwarmEnv:
    def __init__(self, task: str, n: int = 8, seed: int = 0, shared_goal_prob: float = 0.0):
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {TASKS}")
        self.task, self.n = task, n
        self.shared_goal_prob = shared_goal_prob
        scenario, mapping, self.episode_steps = TASK_SETUP[task]
        self.cfg = load_config(None, {
            "drones": {"count": n},
            "sim": {"seed": seed},
            "scenario": {"name": scenario},
            "avoidance": {"method": "none"},     # the policy must avoid things itself
            "mapping": {"enabled": mapping},
            "comms": {"enabled": False},
        })
        self.hold = _Hold()
        self.sim = Simulation(self.cfg, controller=self.hold)
        self.rng = np.random.default_rng(seed + 1000)
        d = self.cfg.drones
        self.observer = Observer(d.max_speed, d.perception_radius, self.sim.maps, self.sim.world.size)
        self.obs_dim = OBS_DIM
        self.episode = 0
        self.reset()

    # ------------------------------------------------------------- helpers
    def _free_point(self) -> np.ndarray:
        w = self.sim.world
        for _ in range(1000):
            p = self.rng.uniform([12.0, 12.0], [w.width - 12.0, w.height - 12.0])
            if w.obstacles.count == 0 or w.obstacles.signed_distance(p[None])[0][0] > 4.0:
                return p
        return w.size / 2

    def _targets(self) -> np.ndarray | None:
        if self.task == "navigate":
            return self.goals
        if self.task == "formation":
            return self._slots()
        return None

    def _slots(self) -> np.ndarray:
        L = self.sim.state.pos[0]
        c, s = math.cos(self.heading), math.sin(self.heading)
        rot = np.array([[c, -s], [s, c]])
        slots = L[None] + self.offsets @ rot.T
        slots[0] = np.nan                       # the leader has no slot
        return slots

    def observe(self) -> np.ndarray:
        st = self.sim.state
        return self.observer(st.pos, st.vel, self._targets(), self.sim.scan, self.sim.maps)

    def global_state(self, obs: np.ndarray) -> np.ndarray:
        """Centralised information for the critic only (MAPPO-style):
        every agent's critic also sees the team-average observation."""
        return np.concatenate([obs, np.repeat(obs.mean(0, keepdims=True), len(obs), 0)], axis=1)

    @property
    def trainable(self) -> np.ndarray:
        """Agents whose experience is used for learning (not the scripted leader)."""
        m = np.ones(self.n, dtype=bool)
        if self.task == "formation":
            m[0] = False
        return m

    # ---------------------------------------------------------------- API
    def reset(self) -> np.ndarray:
        self.episode += 1
        self.sim.reset(seed=int(self.rng.integers(1 << 30)))
        self.t = 0
        if self.task == "navigate":
            self.shared = self.rng.random() < self.shared_goal_prob
            if self.shared:
                self.goals = np.repeat(self._free_point()[None], self.n, 0)
            else:
                self.goals = np.array([self._free_point() for _ in range(self.n)])
            self.prev_dist = np.linalg.norm(self.goals - self.sim.state.pos, axis=1)
            self.arrived_once = np.zeros(self.n, dtype=bool)
        if self.task == "formation":
            self.offsets = slot_offsets("v", self.n, 5.0)
            self.heading = float(self.rng.uniform(-math.pi, math.pi))
            self.leader_wp = self._free_point()
            # followers start roughly in place so the task is "hold", then "track"
            self.sim.state.pos[:] = self._slots_init()
            self.sim.state.vel[:] = 0.0
            self.sim._refresh(0)
            self.prev_err = np.linalg.norm(self.sim.state.pos - self._slots(), axis=1)
            self.prev_err[0] = 0.0
        if self.task == "search":
            self.prev_known = self._known_counts()
        return self.observe()

    def _slots_init(self) -> np.ndarray:
        p = self.sim.state.pos.copy()
        p[0] = self.sim.world.size / 2
        c, s = math.cos(self.heading), math.sin(self.heading)
        slots = p[0][None] + self.offsets @ np.array([[c, -s], [s, c]]).T
        jitter = self.rng.normal(0.0, 3.0, slots.shape)
        jitter[0] = 0.0
        return np.clip(slots + jitter, 5.0, self.sim.world.size - 5.0)

    def _known_counts(self) -> np.ndarray:
        return (np.abs(self.sim.maps.L) >= self.cfg.mapping.known_threshold).sum(1)

    def _leader_action(self) -> np.ndarray:
        """The scripted leader: fly toward a random waypoint at 3.5 m/s."""
        st = self.sim.state
        if np.linalg.norm(self.leader_wp - st.pos[0]) < 8.0:
            self.leader_wp = self._free_point()
        to = self.leader_wp - st.pos[0]
        return to / max(np.linalg.norm(to), 1e-6) * 3.5 / self.cfg.drones.max_speed

    def step(self, actions: np.ndarray):
        """actions (N, 2) in [-1, 1]. Returns obs, rewards, done, info."""
        d = self.cfg.drones
        a = np.clip(actions, -1.0, 1.0)
        if self.task == "formation":
            a = a.copy()
            a[0] = self._leader_action()
        self.hold.accel = action_to_accel(a, self.sim.state.vel, d.max_speed, d.max_accel)
        prev_vel = self.sim.state.vel.copy()
        self.sim.step()
        st = self.sim.state
        self.t += 1

        r = np.zeros(self.n)
        # ---- safety terms (every task)
        if self.n > 1:
            gap = np.linalg.norm(st.pos[:, None] - st.pos[None], axis=2)
            np.fill_diagonal(gap, np.inf)
            nearest = gap.min(1)
            r -= 3.0 * (nearest < 2 * d.radius)                         # collision
            r -= 0.3 * np.clip((3.0 - nearest) / 2.0, 0.0, 1.0)         # near miss
        else:
            nearest = np.full(self.n, np.inf)
        obstacles = self.sim.world.obstacles
        if obstacles.count:
            sd, _ = obstacles.signed_distance(st.pos)
            r -= 3.0 * (sd < d.radius + 0.05)                           # obstacle hit
            r -= 0.2 * np.clip((2.0 - (sd - d.radius)) / 2.0, 0.0, 1.0)
        edge = np.minimum(st.pos, self.sim.world.size - st.pos).min(1)
        r -= 0.2 * np.clip((4.0 - edge) / 4.0, 0.0, 1.0)                # hugging the arena wall
        r -= 0.01 * ((self.sim.state.vel - prev_vel) ** 2).sum(1)      # jerky flying

        # ---- task terms
        info = {}
        if self.task == "navigate":
            dist = np.linalg.norm(self.goals - st.pos, axis=1)
            r += 0.25 * (self.prev_dist - dist)
            arrived = dist < 3.0
            r += 2.0 * arrived
            if self.shared:
                # a shared goal stays put; arrived drones just hold there
                r -= 2.0 * arrived                                   # no repeated bonus for loitering
                r += 2.0 * (arrived & ~self.arrived_once)
                self.arrived_once |= arrived
            else:
                for k in np.flatnonzero(arrived):
                    self.goals[k] = self._free_point()
            self.prev_dist = np.linalg.norm(self.goals - st.pos, axis=1)
            info["arrivals"] = int(arrived.sum())
        elif self.task == "flock":
            heading = st.vel / np.maximum(np.linalg.norm(st.vel, axis=1, keepdims=True), 1e-6)
            nb = self.sim.neighbors
            align = np.zeros(self.n)
            cnt = np.bincount(nb.i, minlength=self.n)
            if len(nb.i):
                np.add.at(align, nb.i, (heading[nb.i] * heading[nb.j]).sum(1))
            align = np.where(cnt > 0, align / np.maximum(cnt, 1), 0.0)
            speed = np.linalg.norm(st.vel, axis=1)
            r += 0.1 * align
            r -= 0.1 * np.abs(speed - d.cruise_speed) / d.cruise_speed
            r -= 0.2 * (cnt == 0)                                       # alone
            r += 0.05 * np.minimum(cnt, 4) / 4                          # stay with the group
            info["alignment"] = float(align.mean())
        elif self.task == "formation":
            self.heading = math.atan2(st.vel[0, 1], st.vel[0, 0]) if np.linalg.norm(st.vel[0]) > 0.5 else self.heading
            err = np.linalg.norm(st.pos - self._slots(), axis=1)
            err[0] = 0.0
            r += 0.25 * (self.prev_err - err)
            self.prev_err = err
            r -= 0.01 * np.clip(err, 0.0, 30.0)
            r += 0.05 * (err < 1.5)
            info["slot_error"] = float(np.nanmean(err[1:])) if self.n > 1 else 0.0
        elif self.task == "search":
            known = self._known_counts()
            r += 0.02 * (known - self.prev_known)
            self.prev_known = known
            info["coverage"] = float(self.sim.info_metrics().map_coverage_mean) if self.t % 100 == 0 else None

        info["collisions"] = int((nearest < 2 * d.radius).sum())
        done = self.t >= self.episode_steps
        return self.observe(), r.astype(np.float32), done, info
