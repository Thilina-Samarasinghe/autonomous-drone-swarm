"""Built-in test scenarios. Each returns obstacles, a spawn region and a goal.

World is 200 x 120 m. Spawn is on the left, goal on the right, so every
obstacle scenario forces the swarm to cross the obstacle field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.sim.obstacles import Obstacles

SCENARIOS = ("open", "forest", "corridor", "random")


@dataclass
class Scenario:
    name: str
    obstacles: Obstacles
    spawn_region: tuple[float, float, float, float] | None   # x0, y0, x1, y1
    goal: np.ndarray | None


def build_scenario(name: str, world_size: np.ndarray, rng: np.random.Generator,
                   num_obstacles: int = 25) -> Scenario:
    w, h = float(world_size[0]), float(world_size[1])
    left_spawn = (15.0, h / 2 - 25, 45.0, h / 2 + 25)
    right_goal = np.array([w - 22.0, h / 2])

    if name == "open":
        return Scenario(name, Obstacles(), None, None)

    if name == "forest":
        # staggered grid of pillars, like flying through a forest
        circles = []
        for col, x in enumerate(np.arange(65.0, w - 55.0, 13.0)):
            offset = 6.5 if col % 2 else 0.0
            for y in np.arange(12.0 + offset, h - 8.0, 13.0):
                circles.append((x, y, 2.0))
        return Scenario(name, Obstacles(circles=circles), left_spawn, right_goal)

    if name == "corridor":
        # a wall across the arena with one 14 m gap, plus two pillars after it
        gap_lo, gap_hi = h / 2 - 7, h / 2 + 7
        boxes = [(95.0, 0.0, 105.0, gap_lo), (95.0, gap_hi, 105.0, h)]
        circles = [(130.0, h / 2 + 12, 3.0), (130.0, h / 2 - 12, 3.0)]
        return Scenario(name, Obstacles(circles=circles, boxes=boxes), left_spawn, right_goal)

    if name == "random":
        circles, boxes = [], []
        placed: list[tuple[float, float, float]] = []   # (x, y, bounding radius)
        tries = 0
        while len(placed) < num_obstacles and tries < 5000:
            tries += 1
            x, y = rng.uniform(60.0, w - 50.0), rng.uniform(8.0, h - 8.0)
            if rng.random() < 0.6:
                r = rng.uniform(1.5, 4.5)
                shape, bound = ("c", (x, y, r)), r
            else:
                hw, hh = rng.uniform(1.5, 6.0), rng.uniform(1.5, 6.0)
                shape, bound = ("b", (x - hw, y - hh, x + hw, y + hh)), float(np.hypot(hw, hh))
            # keep a drone-sized passage (>= 6 m) between obstacles
            if any(np.hypot(x - px, y - py) < bound + pb + 6.0 for px, py, pb in placed):
                continue
            placed.append((x, y, bound))
            (circles if shape[0] == "c" else boxes).append(shape[1])
        return Scenario(name, Obstacles(circles=circles, boxes=boxes), left_spawn, right_goal)

    raise ValueError(f"Unknown scenario '{name}'. Choose from {SCENARIOS}")
