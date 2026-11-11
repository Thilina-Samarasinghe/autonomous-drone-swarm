"""Fixed-timestep, seeded simulation loop. Renderer-independent (runs headless)."""

from __future__ import annotations

import numpy as np

from swarm.agents.sensors import Detections, RangeSensor, Scan
from swarm.agents.state import SwarmState
from swarm.config import Config
from swarm.control.base import Controller, StepContext
from swarm.control.factory import build_controller
from swarm.metrics import Metrics, compute_metrics
from swarm.sim.dynamics import integrate
from swarm.sim.neighbors import NeighborPairs, neighbor_pairs
from swarm.sim.scenarios import Scenario, build_scenario
from swarm.sim.world import World


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
        self.neighbors: NeighborPairs
        self.scan: Scan
        self.detections: Detections
        self.metrics: Metrics
        self._refresh(obstacle_hits=0)

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
        self._refresh(obstacle_hits=0)

    def remove_drones(self, k: int) -> None:
        self.state.remove_last(min(k, self.state.n - 1))
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
        if (self.time_to_arrive_90 is None and self.goal is not None
                and self.metrics.arrived >= 0.9 * self.state.n):
            self.time_to_arrive_90 = self.time
        return self.metrics

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
