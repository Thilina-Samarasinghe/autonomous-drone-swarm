"""Phase 4: cost maps, A*, cost-to-go field, swarm planner, maze."""

import json
import math

import numpy as np
import pytest

from swarm.config import MappingConfig, PlanningConfig, load_config
from swarm.mapping.occupancy import OccupancyMaps
from swarm.planning.grid import GridGraph, astar, cost_map, descend

CFG = PlanningConfig()


def path_cost(cost, path):
    total = 0.0
    for (y0, x0), (y1, x1) in zip(path, path[1:]):
        assert max(abs(y1 - y0), abs(x1 - x0)) == 1                     # 8-connected step
        assert np.isfinite(cost[y1, x1])                                # never through a wall
        if y1 != y0 and x1 != x0:                                       # no corner cutting
            assert np.isfinite(cost[y0, x1]) and np.isfinite(cost[y1, x0])
        length = math.sqrt(2) if (y1 != y0 and x1 != x0) else 1.0
        total += length * 0.5 * (cost[y0, x0] + cost[y1, x1])
    return total


def field_path(cost, start, goal):
    g = GridGraph(*cost.shape)
    w = g.edge_weights(cost)
    field = g.cost_to_go(cost, goal, w)
    return descend(field, g.next_cell(field, w), start), field


# -------------------------------------------------------------- cost map
def test_cost_map_layers():
    L = np.zeros(20 * 20, dtype=np.float32)
    grid = L.reshape(20, 20)
    grid[10, 10] = 4.0                     # one known obstacle
    grid[0:5, :] = -4.0                    # a known-free strip
    c = cost_map(L, 20, 20, CFG, known_thr=1.0, inflate_cells=1)
    assert np.isinf(c[10, 10])
    assert c[10, 11] >= CFG.inflation_cost                               # inflated ring
    assert c[2, 2] == pytest.approx(1.0)                                # known free
    assert c[18, 2] == pytest.approx(1.0 + CFG.unknown_cost)            # unknown, far from obstacle
    assert c[10, 12] > c[10, 16] >= 1.0                                 # clearance cost near obstacles


# ------------------------------------------------------------------ A*
def test_astar_straight_line_in_open_space():
    cost = np.ones((10, 30))
    path = astar(cost, (5, 2), (5, 27))
    assert path[0] == (5, 2) and path[-1] == (5, 27)
    assert path_cost(cost, path) == pytest.approx(25.0)


def test_astar_goes_around_a_wall():
    cost = np.ones((20, 20))
    cost[0:15, 10] = np.inf                # wall with a gap at the bottom
    path = astar(cost, (2, 2), (2, 17))
    assert all(np.isfinite(cost[c]) for c in path)
    assert max(y for y, _ in path) >= 15                                # went through the gap


def test_astar_unreachable_returns_none():
    cost = np.ones((10, 10))
    cost[:, 5] = np.inf
    assert astar(cost, (5, 1), (5, 8)) is None


def test_no_corner_cutting():
    cost = np.ones((3, 3))
    cost[0, 1] = cost[1, 0] = np.inf       # only a diagonal squeeze from (0,0) to (1,1)
    assert astar(cost, (0, 0), (1, 1)) is None
    path, _ = field_path(cost, (0, 0), (1, 1))
    assert path is None


# ---------------------------------------------------------- cost field
def test_field_is_zero_at_goal_and_inf_when_enclosed():
    cost = np.ones((12, 12))
    cost[3:8, 3] = cost[3:8, 7] = cost[3, 3:8] = cost[7, 3:8] = np.inf  # closed box
    _, field = field_path(cost, (0, 0), (10, 10))
    assert field[10, 10] == 0.0
    assert np.isinf(field[5, 5])           # inside the box
    assert np.isfinite(field[0, 0])


@pytest.mark.parametrize("seed", range(6))
def test_field_and_astar_find_equally_short_paths(seed):
    rng = np.random.default_rng(seed)
    cost = 1.0 + rng.random((25, 35)) * 2.0
    cost[rng.random((25, 35)) < 0.22] = np.inf
    start, goal = (1, 1), (23, 33)
    cost[start] = cost[goal] = 1.0
    a = astar(cost, start, goal)
    f, _ = field_path(cost, start, goal)
    if a is None:
        assert f is None
        return
    assert f[0] == start and f[-1] == goal
    assert path_cost(cost, f) == pytest.approx(path_cost(cost, a), rel=1e-9)


# ------------------------------------------------- map revision counter
def test_occupancy_revision_counts_flips():
    maps = OccupancyMaps(2, np.array([40.0, 40.0]), MappingConfig())
    assert maps.occ_revision.tolist() == [0, 0]
    idx = np.array([5, 6])
    maps.merge(0, idx, np.array([3.5, -3.5], dtype=np.float32), relay=True)
    assert maps.occ_revision[0] == 1       # only the occupied cell flips the obstacle layout
    maps.merge(0, idx, np.array([3.5, -3.5], dtype=np.float32), relay=True)
    assert maps.occ_revision[0] == 1       # idempotent merge: no change


# ------------------------------------------------------------- planner
def make_sim(**over):
    from swarm.sim.engine import Simulation
    base = {"scenario": {"name": "maze"}, "planning": {"enabled": True}}
    for k, v in over.items():
        base.setdefault(k, {}).update(v) if isinstance(v, dict) else base.update({k: v})
    return Simulation(load_config(None, base))


def test_planner_replans_when_path_becomes_blocked():
    sim = make_sim(drones={"count": 3})
    sim.run(30)
    pl = sim.planner
    k = 0
    assert pl.paths[k] is not None
    cell = pl.path_cells[k][min(len(pl.path_cells[k]) - 1, pl.progress[k] + 3)]
    sim.maps.L[k, cell] = 4.0              # pretend the drone just saw a wall on its path
    before = pl.stats.blocked_replans
    sim.run(10)
    assert pl.stats.blocked_replans > before
    assert cell not in set(pl.path_cells[k].tolist())


def test_carrot_never_crosses_known_obstacle():
    sim = make_sim(drones={"count": 10})
    pl = sim.planner
    for _ in range(40):
        sim.run(10)
        for k in range(sim.state.n):
            if pl.carrots is None or pl.paths[k] is None:
                continue
            p, c = sim.state.pos[k], pl.carrots[k]
            pts = p + (c - p) * np.linspace(0, 1, 50)[:, None]
            far = np.linalg.norm(pts - p, axis=1) >= 1.5 * pl.cell
            cells = (pts[:, 1] // pl.cell).astype(int) * pl.gw + (pts[:, 0] // pl.cell).astype(int)
            known_occ = sim.maps.L[k, cells] >= sim.cfg.mapping.known_threshold
            assert not (known_occ & far).any()


def test_invalid_algorithm_rejected():
    with pytest.raises(ValueError):
        make_sim(drones={"count": 2}, planning={"enabled": True, "algorithm": "teleport"})


def test_planning_requires_mapping():
    with pytest.raises(ValueError):
        make_sim(drones={"count": 2}, mapping={"enabled": False})


# ---------------------------------------------------------- end to end
def test_swarm_solves_maze_only_with_planning():
    from swarm.sim.engine import Simulation
    results = {}
    for plan in (False, True):
        sim = Simulation(load_config(None, {"drones": {"count": 15}, "scenario": {"name": "maze"},
                                            "planning": {"enabled": plan}}))
        h = sim.run(3200)                   # 160 s
        results[plan] = h[-1].arrived
        assert sim.total_collisions == 0 and sim.total_obstacle_hits == 0
    assert results[False] == 0             # straight-at-goal gets stuck behind the first wall
    assert results[True] >= 14


def test_cli_plan_flags(tmp_path):
    from swarm.main import main
    main(["--headless", "--scenario", "maze", "--drones", "6", "--steps", "200",
          "--planner", "astar", "--out", str(tmp_path)])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["planning"] == "astar" and s["plans"] > 0 and s["failed_plans"] == 0


def test_viewer_planning_toggle(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    from swarm.sim.engine import Simulation
    from swarm.viz.render2d import Viewer
    sim = Simulation(load_config(None, {"drones": {"count": 8}}))
    v = Viewer(sim)
    for key in (pygame.K_6, pygame.K_k):
        v._handle(pygame.event.Event(pygame.KEYDOWN, key=key))
    assert sim.cfg.scenario.name == "maze" and sim.planner is not None
    for _ in range(5):
        v._advance(4)
        v._draw_world(); v._draw_swarm(); v._draw_hud()
    v._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_k))
    assert sim.planner is None
    pygame.quit()
