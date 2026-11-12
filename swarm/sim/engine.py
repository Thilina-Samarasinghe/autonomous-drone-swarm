"""Fixed-timestep, seeded simulation loop. Renderer-independent (runs headless)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.agents.sensors import Detections, RangeSensor, Scan
from swarm.agents.state import SwarmState
from swarm.config import Config
from swarm.control.base import Controller, StepContext
from swarm.comms.network import CommsNetwork
from swarm.control.factory import build_controller
from swarm.mapping.occupancy import GroundTruth, OccupancyMaps
from swarm.formation.manager import FormationManager
from swarm.mission.manager import RescueMission
from swarm.terrain.follow import RouteFollower
from swarm.terrain.route_planner import plan_routes
from swarm.formation.shapes import SHAPES
from swarm.planning.planner import SwarmPlanner
from swarm.metrics import Metrics, compute_metrics
from swarm.sim.dynamics import integrate
from swarm.sim.neighbors import NeighborPairs, neighbor_pairs
from swarm.sim.scenarios import Scenario, build_scenario
from swarm.sim.world import World


@dataclass
class InfoMetrics:
    """What the swarm knows (Phase 3). Computed on demand: it is heavier than Metrics."""
    time: float
    map_coverage_mean: float     # avg fraction of the arena each drone has mapped
    map_coverage_min: float      # the least-informed drone
    map_coverage_team: float     # union of all drones' maps
    map_accuracy: float          # known cells classified correctly
    comm_links: int
    comm_groups: int             # connected components of the radio graph
    teammates_known: float       # avg fraction of other drones each drone has heard of
    info_age: float              # avg age (s) of what drones know about each other
    delivered: int
    dropped: int
    kbytes_sent: float


class Simulation:
    def __init__(self, cfg: Config, controller: Controller | None = None):
        self.cfg = cfg
        self._custom_controller = controller
        self.sensor = RangeSensor(cfg.sensor.num_rays, cfg.sensor.range)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.cfg.sim.seed if seed is None else seed)
        base = World.from_config(self.cfg.world)
        sc = self.cfg.scenario
        mc = self.cfg.mission
        self.scenario: Scenario = build_scenario(sc.name, base.size, self.rng, sc.num_obstacles,
                                                 mc.victims, (mc.base_x, mc.base_y),
                                                 self.cfg.drones.count, self.cfg.terrain)
        self.world = World.from_config(self.cfg.world, self.scenario.obstacles)
        self.goal: np.ndarray | None = None if self.scenario.goal is None else self.scenario.goal.copy()
        # the rescue mission runs in scenarios that contain victims
        self.mission_active = (mc.enabled and self.scenario.victims is not None
                               and len(self.scenario.victims) > 0)
        self.terrain = self.scenario.terrain
        self.route_plan = None
        if self.terrain is not None:
            # pre-flight route planning for every drone (start -> its own goal)
            tc = self.cfg.terrain
            self.route_plan = plan_routes(tc.planner, self.scenario.starts, self.scenario.goals,
                                          self.terrain, tc, base.size, 8.0,
                                          np.random.default_rng([self.cfg.sim.seed, 13]))
        d = self.cfg.drones
        if self.scenario.starts is not None:
            starts = self.scenario.starts
            self.state = SwarmState(starts.copy(), np.zeros_like(starts), np.arange(len(starts)))
        else:
            self.state = SwarmState.spawn(d.count, self.world, self.rng, d.spawn_spacing,
                                          d.spawn_speed, region=self.scenario.spawn_region)
        self.time = 0.0
        self.step_count = 0
        self.total_collisions = 0
        self.total_obstacle_hits = 0
        self.time_to_arrive_90: float | None = None
        self.time_leader_arrived: float | None = None   # formation runs: drone 0 reached the goal
        self._init_info_systems()
        self.controller: Controller = self._make_controller()
        self.neighbors: NeighborPairs
        self.scan: Scan
        self.detections: Detections
        self.metrics: Metrics
        self._refresh(obstacle_hits=0)

    def _init_info_systems(self) -> None:
        """(Re)create maps and radio for the current drone count."""
        n = self.state.n
        self.maps = (OccupancyMaps(n, self.world.size, self.cfg.mapping)
                     if self.cfg.mapping.enabled else None)
        self.truth = GroundTruth(self.maps, self.world.obstacles) if self.maps is not None else None
        # separate random stream so radio randomness never changes the flight
        comms_rng = np.random.default_rng([self.cfg.sim.seed, 7])
        self.comms = (CommsNetwork(n, self.cfg.comms, self.cfg.sim.dt, comms_rng)
                      if self.cfg.comms.enabled else None)
        self._map_every = max(1, round(self.cfg.mapping.update_interval / self.cfg.sim.dt))
        # fault injection (Phase 9): failed drones land and go silent
        self.alive = np.ones(n, dtype=bool)
        self.failed_at: float | None = None
        self._dead_pos = np.zeros((n, 2))
        self._fault_rng = np.random.default_rng([self.cfg.sim.seed, 17])
        self.follower = None
        if self.route_plan is not None and n == len(self.route_plan.routes):
            self.follower = RouteFollower(self.route_plan.routes, self.terrain, self.cfg.terrain.speed,
                                          delays=self.route_plan.delays,
                                          separation=self.cfg.terrain.separation)
        self.formation = None
        if self.cfg.formation.enabled and not self.mission_active and self.terrain is None:
            if self.cfg.formation.shape not in SHAPES:
                raise ValueError(f"Unknown formation shape '{self.cfg.formation.shape}'. Choose from {SHAPES}")
            self.formation = FormationManager(n, self.cfg.formation, self.cfg.sim.dt, self.world.size)
        self.mission = None
        if self.mission_active:
            self.mission = RescueMission(n, self.scenario.victims, self.scenario.base, self.cfg.mission,
                                         self.world.size, self.cfg.sim.dt, self.cfg.comms.range,
                                         np.random.default_rng([self.cfg.sim.seed, 11]),
                                         edge_margin=self.world.boundary_margin + 2.0)
        self.planner = None
        wants_planner = self.cfg.planning.enabled or (self.mission_active and self.cfg.mission.use_planner)
        if wants_planner:
            if self.maps is None:
                raise ValueError("planning needs mapping.enabled = true (drones plan on their own maps)")
            self.planner = SwarmPlanner(n, self.maps, self.cfg.planning, self.cfg.mapping,
                                        self.cfg.drones, self.cfg.sim.dt,
                                        self.cfg.avoidance.safety_margin)

    def set_planning(self, enabled: bool) -> None:
        """Switch path planning on/off on the fly (keeps the swarm and its maps)."""
        self.cfg.planning.enabled = enabled
        if enabled and self.maps is not None:
            self.planner = SwarmPlanner(self.state.n, self.maps, self.cfg.planning, self.cfg.mapping,
                                        self.cfg.drones, self.cfg.sim.dt,
                                        self.cfg.avoidance.safety_margin)
        else:
            self.planner = None

    def set_formation(self, shape: str | None) -> None:
        """Fly in formation `shape`, or flock freely with None (keeps the swarm state)."""
        if shape is None:
            self.cfg.formation.enabled = False
            self.formation = None
        elif self.formation is not None:
            self.cfg.formation.shape = shape
            self.formation.set_shape(shape)
        else:
            self.cfg.formation.enabled = True
            self.cfg.formation.shape = shape
            self.formation = FormationManager(self.state.n, self.cfg.formation, self.cfg.sim.dt,
                                              self.world.size)
        self.controller = self._make_controller()

    def set_avoidance(self, method: str) -> None:
        """Switch avoidance method on the fly (keeps the current swarm state)."""
        self.cfg.avoidance.method = method
        self.controller = self._make_controller()

    def _make_controller(self) -> Controller:
        """The controller stack for the current settings: a custom controller if
        one was given, a trained policy if `rl.policy` is set, else the
        hand-written controllers."""
        if self._custom_controller is not None:
            return self._custom_controller
        if self.cfg.rl.policy:
            from swarm.control.orca import OrcaSafetyLayer
            from swarm.rl.controller import RLController
            d = self.cfg.drones
            ctrl = RLController(self.cfg.rl.policy, d.max_speed, d.max_accel, d.perception_radius,
                                self.maps, self.world.size)
            return OrcaSafetyLayer(ctrl, d, self.cfg.avoidance) if self.cfg.rl.orca else ctrl
        return build_controller(self.cfg, mission=self.mission_active or self.terrain is not None)

    def add_drones(self, k: int) -> None:
        d = self.cfg.drones
        c = self.world.size / 2 if self.scenario.spawn_region is None else np.array(
            [(self.scenario.spawn_region[0] + self.scenario.spawn_region[2]) / 2,
             (self.scenario.spawn_region[1] + self.scenario.spawn_region[3]) / 2])
        pos = c + self.rng.normal(0, 5, (k, 2))
        heading = self.rng.uniform(0, 2 * np.pi, k)
        vel = np.stack([np.cos(heading), np.sin(heading)], 1) * d.spawn_speed
        self.state.add(pos, vel)
        self._init_info_systems()   # maps and radio are sized per drone, so they restart
        self.controller = self._make_controller()
        self._refresh(obstacle_hits=0)

    def remove_drones(self, k: int) -> None:
        self.state.remove_last(min(k, self.state.n - 1))
        self._init_info_systems()
        self.controller = self._make_controller()
        self._refresh(obstacle_hits=0)

    def _refresh(self, obstacle_hits: int) -> None:
        d = self.cfg.drones
        self.neighbors = neighbor_pairs(self.state.pos, d.perception_radius)
        alive = getattr(self, "alive", None)
        if alive is not None and not alive.all() and len(self.neighbors.i):
            # landed drones are not sensed as neighbours (they are on the ground)
            keep = alive[self.neighbors.i] & alive[self.neighbors.j]
            nb = self.neighbors
            self.neighbors = type(nb)(nb.i[keep], nb.j[keep], nb.diff[keep], nb.dist[keep])
        self.scan = self.sensor.scan(self.state.pos, self.world)
        self._degrade_lidar()
        self.detections = self.scan.closest_points()
        if alive is not None and not alive.all():
            # metrics describe the flying swarm only
            idx = np.flatnonzero(alive)
            remap = -np.ones(self.state.n, dtype=int)
            remap[idx] = np.arange(len(idx))
            nb = self.neighbors
            sub_nb = type(nb)(remap[nb.i], remap[nb.j], nb.diff, nb.dist)
            sub = SwarmState(self.state.pos[idx], self.state.vel[idx], self.state.ids[idx])
            self.metrics = compute_metrics(sub, sub_nb, d.radius, self.time, self.world,
                                           self.goal, self.cfg.scenario.goal_radius, obstacle_hits)
        else:
            self.metrics = compute_metrics(
                self.state, self.neighbors, d.radius, self.time, self.world,
                self.goal, self.cfg.scenario.goal_radius, obstacle_hits)

    def _degrade_lidar(self) -> None:
        f = self.cfg.faults
        if f.lidar_noise <= 0 and f.lidar_dropout <= 0:
            return
        rng = getattr(self, "_fault_rng", None) or np.random.default_rng(0)
        dist = self.scan.dist
        hit = dist < self.scan.max_range - 1e-9
        if f.lidar_noise > 0:
            noisy = dist + rng.normal(0.0, f.lidar_noise, dist.shape)
            dist = np.where(hit, np.clip(noisy, 0.0, self.scan.max_range - 1e-6), dist)
        if f.lidar_dropout > 0:
            dist = np.where(rng.random(dist.shape) < f.lidar_dropout, self.scan.max_range, dist)
        self.scan.dist = dist

    def fail(self, ids) -> None:
        """Drones `ids` fail: they land where they are and go silent (no sensing,
        radio, avoidance or mission role). A failed formation leader is replaced."""
        ids = np.atleast_1d(np.asarray(ids, dtype=int))
        ids = ids[self.alive[ids]]
        if len(ids) == 0:
            return
        self.alive[ids] = False
        self._dead_pos[ids] = self.state.pos[ids]
        self.state.vel[ids] = 0.0
        if self.mission is not None:
            from swarm.mission.manager import DEAD
            self.mission.mode[ids] = DEAD
        if self.formation is not None:
            self.formation.set_alive(self.alive)
        self._refresh(obstacle_hits=0)

    def _scheduled_faults(self) -> None:
        f = self.cfg.faults
        if f.fail_time < 0 or self.failed_at is not None or self.time < f.fail_time:
            return
        self.failed_at = self.time
        n = self.state.n
        candidates = np.flatnonzero(self.alive)
        if not f.fail_leader:
            candidates = candidates[candidates != 0]
        k = int(round(f.fail_fraction * n))
        ids = list(self._fault_rng.choice(candidates, size=min(k, len(candidates)), replace=False))
        if f.fail_leader:
            ids.append(0)
        self.fail(ids)

    # ------------------------------------------------------------------- loop
    def step(self) -> Metrics:
        d = self.cfg.drones
        self._scheduled_faults()
        targets = None
        if self.mission is not None:
            self.mission.update(self.step_count, self.time, self.state.pos, self.state.vel,
                                self.maps, self.comms)
            goals = self.mission.nav_goals()
            if self.planner is not None:
                targets = self.planner.update(self.step_count, self.state.pos, goals, self.maps, 4.0)
                # a target the drone's own map shows no route to: pick another one
                for k in np.flatnonzero(~np.isnan(goals[:, 0])):
                    if (self.planner.paths[k] is None
                            and np.linalg.norm(self.planner.plan_goal[k] - goals[k]) < 1e-6):
                        self.mission.unreachable(k)
        elif self.planner is not None:
            targets = self.planner.update(self.step_count, self.state.pos, self.goal, self.maps,
                                          self.cfg.scenario.goal_radius)
        ctx = StepContext(self.world, self.neighbors, self.goal, self.cfg.sim.dt,
                          self.scan, self.detections, targets=targets)
        if self.mission is not None:
            ctx.extras["desired_velocity"] = self.mission.desired_velocity(self.step_count, self.state.pos,
                                                                           targets, self.time)
        if self.follower is not None:
            ctx.extras["desired_velocity"] = self.follower.desired_velocity(self.state.pos, self.time)
        self.last_formation_targets = None
        if self.formation is not None:
            ctx.extras["formation"] = self.last_formation_targets = self.formation.update(
                self.time, self.state.pos, self.state.vel, self.scan, self.goal, targets,
                self.comms, d.perception_radius, self.maps,
                self.planner.paths if self.planner is not None else None,
                self.planner.progress if self.planner is not None else None)
        accel = self.controller.compute(self.state, ctx)
        dead = ~self.alive
        if dead.any():
            accel[dead] = 0.0
        _, hit = integrate(self.state, accel, self.cfg.sim.dt, d.max_speed, d.max_accel,
                           self.world, d.radius)
        if dead.any():                                  # landed: they stay put
            self.state.pos[dead] = self._dead_pos[dead]
            self.state.vel[dead] = 0.0
            hit &= ~dead

        self.time += self.cfg.sim.dt
        self.step_count += 1
        self._refresh(obstacle_hits=int(hit.sum()))
        self.total_collisions += self.metrics.collisions
        self.total_obstacle_hits += self.metrics.obstacle_hits
        if self.maps is not None and self.step_count % self._map_every == 0:
            if dead.any():                              # a landed drone's map stops changing
                frozen = (self.maps.L[dead].copy(), self.maps.dirty[dead].copy())
            self.maps.integrate_scan(self.state.pos, self.scan)
            if dead.any():
                self.maps.L[dead], self.maps.dirty[dead] = frozen
        if self.comms is not None:
            self.comms.step(self.step_count, self.time, self.state.pos, self.maps, self.state.vel,
                            self.alive)
        if self.follower is not None:
            self.follower.record(self.time, self.state.pos, self.state.vel, self.cfg.sim.dt)
        if (self.time_to_arrive_90 is None and self.goal is not None
                and self.metrics.arrived >= 0.9 * int(self.alive.sum())):
            self.time_to_arrive_90 = self.time
        if (self.time_leader_arrived is None and self.goal is not None and self.state.n
                and np.linalg.norm(self.state.pos[0] - self.goal) < self.cfg.scenario.goal_radius):
            self.time_leader_arrived = self.time
        return self.metrics

    def info_metrics(self) -> InfoMetrics:
        thr = self.cfg.mapping.known_threshold
        if self.maps is not None:
            cov = self.truth.coverage(self.maps.L, thr)
            team = float(self.truth.coverage(self.maps.team_map(), thr))
            acc = self.truth.accuracy(self.maps.L, thr)
            cmean, cmin = float(cov.mean()), float(cov.min())
        else:
            cmean = cmin = team = acc = float("nan")
        if self.comms is not None:
            frac, age = self.comms.knowledge(self.time)
            st = self.comms.stats
            links, groups = len(self.comms.links), self.comms.groups()
            delivered, dropped, kb = st.delivered, st.dropped, st.bytes_sent / 1000
        else:
            frac, age, links, groups, delivered, dropped, kb = 0.0, float("nan"), 0, self.state.n, 0, 0, 0.0
        return InfoMetrics(self.time, cmean, cmin, team, acc, links, groups, frac, age,
                           delivered, dropped, kb)

    def run(self, steps: int, recorder: "Recorder | None" = None) -> list[Metrics]:
        history = []
        for _ in range(steps):
            m = self.step()
            history.append(m)
            if recorder is not None:
                recorder.capture(self)
        return history


class Recorder:
    """Stores positions every `every` steps for offline plots (fixed drone count)."""

    def __init__(self, every: int = 5):
        self.every = every
        self.times: list[float] = []
        self.positions: list[np.ndarray] = []

    def capture(self, sim: Simulation) -> None:
        if sim.step_count % self.every == 0:
            self.times.append(sim.time)
            self.positions.append(sim.state.pos.copy())

    def save(self, path: str) -> None:
        np.savez_compressed(path, times=np.array(self.times), positions=np.stack(self.positions))
