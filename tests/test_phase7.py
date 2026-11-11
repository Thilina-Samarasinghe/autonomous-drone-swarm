"""Phase 7: terrain, threat zones, route cost, PSO / hybrid / grid route planners, flying routes."""

import json
import math

import numpy as np
import pytest

from swarm.config import TerrainConfig, load_config
from swarm.sim.scenarios import build_scenario
from swarm.terrain.route_cost import RouteCost, positions_at_times, resample
from swarm.terrain.route_planner import GridRoutePlanner, plan_routes, pso_route
from swarm.terrain.terrain import Terrain, generate_terrain

W = np.array([200.0, 120.0])


def flat_with_threat(r=20.0):
    return Terrain(hills=np.empty((0, 4)), threats=np.array([[100.0, 60.0, r]]), ceiling=70.0,
                   lethal_fraction=0.35)


# ---------------------------------------------------------------- terrain
def test_height_risk_and_lethal_core():
    t = Terrain(np.array([[50.0, 50.0, 80.0, 10.0]]), np.array([[150.0, 60.0, 20.0]]), 70.0, 0.35)
    assert t.height(np.array([50.0, 50.0])) == pytest.approx(80.0)
    assert t.height(np.array([50.0, 60.0])) == pytest.approx(80.0 * math.exp(-0.5))
    risk, lethal = t.risk(np.array([[150.0, 60.0], [160.0, 60.0], [170.0, 60.0], [155.0, 60.0]]))
    assert risk[0] == pytest.approx(1.0) and risk[1] == pytest.approx(0.25) and risk[2] == 0.0
    assert lethal.tolist() == [True, False, False, True]          # core radius 7 m
    _, grown = t.risk(np.array([[158.0, 60.0]]), lethal_margin=2.0)
    assert grown[0]


def test_no_fly_circle_edge_is_at_the_ceiling():
    t = Terrain(np.array([[50.0, 50.0, 100.0, 12.0]]), np.empty((0, 3)), 70.0, 0.35)
    cx, cy, r = t.no_fly_circles()[0]
    assert t.height(np.array([cx + r, cy])) == pytest.approx(70.0, rel=1e-6)
    low = Terrain(np.array([[50.0, 50.0, 60.0, 12.0]]), np.empty((0, 3)), 70.0, 0.35)
    assert len(low.no_fly_circles()) == 0                          # below the ceiling: flyable


def test_lookup_tables_match_analytic_values():
    t = generate_terrain(W, np.random.default_rng(0), 9, 100.0, 70.0, 8, (10, 20), 0.35,
                         np.array([[16.0, 60.0], [184.0, 60.0]]))
    tab = t.tables(W, 3.0)
    pts = np.random.default_rng(1).uniform([0, 0], W, (5000, 2))
    iy, ix = tab.index(pts)
    assert np.abs(tab.height[iy, ix] - t.height(pts)).max() < 5.0   # < the 5 m ceiling buffer
    assert t.tables(W, 3.0) is tab                                  # cached


def test_generated_terrain_keeps_starts_and_goals_safe():
    sc = build_scenario("terrain", W, np.random.default_rng(5), num_drones=10)
    pts = np.vstack([sc.starts, sc.goals])
    assert (sc.terrain.height(pts) < sc.terrain.ceiling).all()
    assert not sc.terrain.risk(pts)[1].any()
    assert len(sc.starts) == len(sc.goals) == 10
    assert sc.obstacles.count == len(sc.terrain.no_fly_circles())


def test_departure_delay_resolves_a_head_on_conflict():
    """Two drones on the same line in opposite directions: PSO can wait instead of detouring."""
    from swarm.terrain.route_cost import positions_at_times
    poly = np.array([[[0.0, 0.0], [10.0, 0.0]]])
    at = positions_at_times(poly, np.array([0.0, 2.0, 3.0]), speed=5.0, delay=2.0)
    assert np.allclose(at[0], [[0, 0], [0, 0], [5, 0]])               # waits 2 s, then flies


# ------------------------------------------------------------- geometry
def test_resample_and_timing():
    poly = np.array([[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]])
    pts, length = resample(poly, 5)
    assert length[0] == pytest.approx(20.0)
    assert np.allclose(pts[0], [[0, 0], [5, 0], [10, 0], [10, 5], [10, 10]])
    at = positions_at_times(poly, np.array([0.0, 1.0, 3.0, 10.0]), speed=5.0)
    assert np.allclose(at[0], [[0, 0], [5, 0], [10, 5], [10, 10]])  # waits at the goal


# ----------------------------------------------------------------- cost
def test_cost_forbids_lethal_core_and_prefers_safe_detour():
    cfg = TerrainConfig()
    cost = RouteCost(flat_with_threat(), cfg, W, 8.0)
    through = np.array([[[20.0, 60.0], [180.0, 60.0]]])
    around = np.array([[[20.0, 60.0], [100.0, 95.0], [180.0, 60.0]]])
    a, b = cost(through), cost(around)
    assert a.lethal[0] > 0 and b.lethal[0] == 0
    assert b.total[0] < a.total[0]


def test_conflict_counts_same_time_only_and_not_at_endpoints():
    cfg = TerrainConfig(separation=6.0)
    empty = Terrain(np.empty((0, 4)), np.empty((0, 3)), 70.0, 0.35)
    cost = RouteCost(empty, cfg, W, 8.0)
    cost.add_planned(np.array([[60.0, 60.0], [180.0, 60.0]]))          # west -> east, at (100, 60) at t = 8 s
    crossing = np.array([[[100.0, 20.0], [100.0, 100.0]]])            # also at (100, 60) at t = 8 s
    late = np.array([[[100.0, 40.0], [100.0, 100.0]]])                # same line, but there at t = 4 s
    assert cost(crossing).conflict[0] > 0
    assert cost(late).conflict[0] == 0
    # take-off side by side (both near their own starts) is excused ...
    side_by_side = np.array([[[60.0, 64.0], [56.0, 64.0]]])           # parks west, the other flies east
    assert cost(side_by_side).conflict[0] == 0
    # ... but parking where the other drone later flies past is not
    in_the_way = np.array([[[60.0, 64.0], [100.0, 63.0]]])
    assert cost(in_the_way).conflict[0] > 0


# -------------------------------------------------------------- planners
def test_pso_finds_a_safe_route_and_never_gets_worse():
    cfg = TerrainConfig(iterations=80)
    cost = RouteCost(flat_with_threat(25.0), cfg, W, 8.0)
    route, delay, hist = pso_route(np.array([20.0, 60.0]), np.array([180.0, 60.0]), cost, cfg,
                                   np.random.default_rng(0))
    s = cost(route[None], np.array([delay]))
    assert delay == pytest.approx(0.0, abs=0.5)                       # alone: no reason to wait
    assert s.lethal[0] == 0 and s.outside[0] == 0
    assert np.all(np.diff(hist) <= 1e-9)                               # global best is monotone


def test_hybrid_is_never_worse_than_its_grid_seed():
    cfg = TerrainConfig(iterations=40)
    sc = build_scenario("terrain", W, np.random.default_rng(3), num_drones=1, terrain_cfg=cfg)
    cost = RouteCost(sc.terrain, cfg, W, 8.0)
    s0, g0 = sc.starts[0], sc.goals[0]
    seed = GridRoutePlanner(sc.terrain, cfg, W, 8.0).route_with_lanes(s0, g0, cfg.lane)
    route, delay, _ = pso_route(s0, g0, cost, cfg, np.random.default_rng(0), seed)
    # the seed exactly as PSO encodes it: start, lane, K waypoints along the grid route, lane, goal
    from swarm.terrain.route_cost import resample as rs
    pts, _ = rs(seed[1:-1][None], cfg.waypoints + 2)
    seed_poly = np.vstack([seed[:2], pts[0, 1:-1], seed[-2:]])[None]
    assert cost(route[None], np.array([delay])).total[0] <= cost(seed_poly).total[0] + 1e-6


def test_grid_route_is_legal_and_simplified():
    cfg = TerrainConfig()
    sc = build_scenario("terrain", W, np.random.default_rng(7), num_drones=4, terrain_cfg=cfg)
    plan = plan_routes("grid", sc.starts, sc.goals, sc.terrain, cfg, W, 8.0, np.random.default_rng(0))
    for r, s in zip(plan.routes, plan.scores):
        assert s["lethal"] == 0 and s["too_high"] == 0
        assert len(r) < 40                                             # Douglas-Peucker removed the staircase


def test_hybrid_deconflicts_better_than_grid():
    cfg = TerrainConfig(iterations=80)
    sc = build_scenario("terrain", W, np.random.default_rng(11), num_drones=6, terrain_cfg=cfg)
    plans = {m: plan_routes(m, sc.starts, sc.goals, sc.terrain, cfg, W, 8.0, np.random.default_rng(0))
             for m in ("grid", "hybrid")}
    conflict = {m: sum(s["conflict"] for s in p.scores) for m, p in plans.items()}
    assert conflict["hybrid"] < conflict["grid"]
    assert all(s["lethal"] == 0 for s in plans["hybrid"].scores)


def test_unknown_planner_rejected():
    with pytest.raises(ValueError):
        plan_routes("teleport", np.zeros((1, 2)), np.ones((1, 2)), flat_with_threat(), TerrainConfig(),
                    W, 8.0, np.random.default_rng(0))


# ------------------------------------------------------------ end to end
def terrain_sim(n=6, **terrain):
    from swarm.sim.engine import Simulation
    t = {"iterations": 60}
    t.update(terrain)
    return Simulation(load_config(None, {"drones": {"count": n}, "scenario": {"name": "terrain"},
                                         "terrain": t}))


def test_drones_fly_their_routes_safely():
    sim = terrain_sim(6)
    assert np.allclose(sim.state.pos, sim.scenario.starts)
    sim.run(2000)                                                      # 100 s
    fl = sim.follower.stats
    assert np.isfinite(fl.arrival).all()
    assert fl.lethal_time.sum() == 0
    assert sim.total_collisions == 0 and sim.total_obstacle_hits == 0


def test_follower_waits_for_its_departure_slot_and_counts_close_encounters():
    from swarm.terrain.follow import RouteFollower
    t = Terrain(np.empty((0, 4)), np.empty((0, 3)), 70.0, 0.35)
    routes = [np.array([[20.0, 60.0], [180.0, 60.0]]), np.array([[20.0, 64.0], [180.0, 64.0]])]
    f = RouteFollower(routes, t, speed=5.0, delays=[0.0, 4.0], separation=6.0, endpoint_zone=8.0)
    pos = np.array([[100.0, 60.0], [100.0, 64.0]])                  # en route, 4 m apart
    v = f.desired_velocity(pos, time=1.0)
    assert np.linalg.norm(v[0]) > 0 and np.allclose(v[1], 0.0)      # drone 1 still waiting
    f.record(1.0, pos, np.zeros((2, 2)), 0.5)
    assert f.stats.close_pair_seconds == pytest.approx(0.5)
    f.record(1.5, np.array([[21.0, 60.0], [21.0, 64.0]]), np.zeros((2, 2)), 0.5)
    assert f.stats.close_pair_seconds == pytest.approx(0.5)         # near their starts: not counted


def test_straight_routes_fly_into_danger():
    sim = terrain_sim(6, planner="straight")
    sc = sim.route_plan.scores
    assert sum(s["lethal"] + s["too_high"] for s in sc) > 0           # planned through forbidden areas


def test_cli_terrain_flags(tmp_path):
    from swarm.main import main
    main(["--headless", "--scenario", "terrain", "--drones", "4", "--steps", "200",
          "--route-planner", "grid", "--out", str(tmp_path)])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["route_planner"] == "grid" and s["planned_forbidden_m"] == 0
    assert "flown_lethal_s" in s and "flown_above_ceiling_s" in s


def test_viewer_terrain_keys(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    from swarm.sim.engine import Simulation
    from swarm.viz.render2d import Viewer
    sim = Simulation(load_config(None, {"drones": {"count": 4}, "terrain": {"iterations": 20}}))
    v = Viewer(sim)
    v._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_8))
    assert sim.terrain is not None and sim.cfg.terrain.planner == "hybrid"
    v._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_u))
    assert sim.cfg.terrain.planner == "pso" and sim.route_plan is not None
    v._advance(3)
    v._draw_world(); v._draw_swarm(); v._draw_hud()
    pygame.quit()
