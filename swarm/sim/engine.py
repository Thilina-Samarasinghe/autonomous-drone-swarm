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
        self.scenario: Scenario = build_scenario(sc.name, base.size, self.rng, sc.num_obstacles)
        self.world = World.from_config(self.cfg.world, self.scenario.obstacles)
        self.goal: np.ndarray | None = None if self.scenario.goal is None else self.scenario.goal.copy()
        self.controller: Controller = self._custom_controller or build_controller(self.cfg)

        d = self.cfg.drones
        self.state = SwarmState.spawn(d.count, self.world, self.rng, d.spawn_spacing,
                                      d.spawn_speed, region=self.scenario.spawn_region)
        self.time = 0.0
        self.step_count = 0
        self.total_collisions = 0
        self.total_obstacle_hits = 0
        self.time_to_arrive_90: float | None = None
        self._init_info_systems()
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

    def set_avoidance(self, method: str) -> None:
        """Switch avoidance method on the fly (keeps the current swarm state)."""
        self.cfg.avoidance.method = method
        self.controller = build_controller(self.cfg)

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
        self._refresh(obstacle_hits=0)

    def remove_drones(self, k: int) -> None:
        self.state.remove_last(min(k, self.state.n - 1))
        self._init_info_systems()
        self._refresh(obstacle_hits=0)

    def _refresh(self, obstacle_hits: int) -> None:
        d = self.cfg.drones
        self.neighbors = neighbor_pairs(self.state.pos, d.perception_radius)
        self.scan = self.sensor.scan(self.state.pos, self.world)
        self.detections = self.scan.closest_points()
        self.metrics = compute_metrics(
            self.state, self.neighbors, d.radius, self.time, self.world,
            self.goal, self.cfg.scenario.goal_radius, obstacle_hits)

    # ------------------------------------------------------------------- loop
    def step(self) -> Metrics:
        d = self.cfg.drones
        ctx = StepContext(self.world, self.neighbors, self.goal, self.cfg.sim.dt,
                          self.scan, self.detections)
        accel = self.controller.compute(self.state, ctx)
        _, hit = integrate(self.state, accel, self.cfg.sim.dt, d.max_speed, d.max_accel,
                           self.world, d.radius)

        self.time += self.cfg.sim.dt
        self.step_count += 1
        self._refresh(obstacle_hits=int(hit.sum()))
        self.total_collisions += self.metrics.collisions
        self.total_obstacle_hits += self.metrics.obstacle_hits
        if self.maps is not None and self.step_count % self._map_every == 0:
            self.maps.integrate_scan(self.state.pos, self.scan)
        if self.comms is not None:
            self.comms.step(self.step_count, self.time, self.state.pos, self.maps)
        if (self.time_to_arrive_90 is None and self.goal is not None
                and self.metrics.arrived >= 0.9 * self.state.n):
            self.time_to_arrive_90 = self.time
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
