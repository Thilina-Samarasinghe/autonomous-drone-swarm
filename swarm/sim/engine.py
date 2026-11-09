"""Fixed-timestep, seeded simulation loop. Renderer-independent (runs headless)."""

from __future__ import annotations

import numpy as np

from swarm.agents.state import SwarmState
from swarm.config import Config
from swarm.control.base import Controller, StepContext
from swarm.control.boids import BoidsController
from swarm.metrics import Metrics, compute_metrics
from swarm.sim.dynamics import integrate
from swarm.sim.neighbors import NeighborPairs, neighbor_pairs
from swarm.sim.world import World


class Simulation:
    def __init__(self, cfg: Config, controller: Controller | None = None):
        self.cfg = cfg
        self.world = World.from_config(cfg.world)
        self.controller: Controller = controller or BoidsController(cfg.boids, cfg.drones)
        self.goal: np.ndarray | None = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.cfg.sim.seed if seed is None else seed)
        d = self.cfg.drones
        self.state = SwarmState.spawn(d.count, self.world, self.rng, d.spawn_spacing, d.spawn_speed)
        self.time = 0.0
        self.step_count = 0
        self.total_collisions = 0
        self.neighbors: NeighborPairs
        self.metrics: Metrics
        self._refresh()

    def add_drones(self, k: int) -> None:
        d = self.cfg.drones
        c = self.world.size / 2
        pos = c + self.rng.normal(0, 5, (k, 2))
        heading = self.rng.uniform(0, 2 * np.pi, k)
        vel = np.stack([np.cos(heading), np.sin(heading)], 1) * d.cruise_speed
        self.state.add(pos, vel)
        self._refresh()

    def remove_drones(self, k: int) -> None:
        self.state.remove_last(min(k, self.state.n - 1))
        self._refresh()

    def _refresh(self) -> None:
        d = self.cfg.drones
        self.neighbors = neighbor_pairs(self.state.pos, d.perception_radius)
        self.metrics = compute_metrics(self.state, self.neighbors, d.radius, self.time)

    # ------------------------------------------------------------------- loop
    def step(self) -> Metrics:
        d = self.cfg.drones
        ctx = StepContext(self.world, self.neighbors, self.goal)
        accel = self.controller.compute(self.state, ctx)
        integrate(self.state, accel, self.cfg.sim.dt, d.max_speed, d.max_accel, self.world, d.radius)

        self.time += self.cfg.sim.dt
        self.step_count += 1
        self._refresh()
        self.total_collisions += self.metrics.collisions
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
