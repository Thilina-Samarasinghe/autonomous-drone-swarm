"""Phase 2: obstacles, lidar, APF, ORCA, scenarios."""

import json

import numpy as np
import pytest

from swarm.agents.sensors import RangeSensor
from swarm.agents.state import SwarmState
from swarm.config import load_config
from swarm.control.base import StepContext
from swarm.control.boids import BoidsController
from swarm.control.factory import METHODS, build_controller
from swarm.control.orca import OrcaSafetyLayer, _lp2
from swarm.sim.neighbors import neighbor_pairs
from swarm.sim.obstacles import Obstacles
from swarm.sim.scenarios import SCENARIOS, build_scenario
from swarm.sim.world import World

DIRS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])


def cfg_for(**over):
    return load_config(None, over)


# ------------------------------------------------------------- geometry
def test_signed_distance_circle_and_box():
    obs = Obstacles(circles=[(10, 10, 2)], boxes=[(20, 0, 30, 10)])
    d, n = obs.signed_distance(np.array([[15.0, 10.0], [25.0, 5.0], [10.0, 13.0], [17.0, 5.0]]))
    assert d[0] == pytest.approx(3.0)            # 5 from circle centre minus radius 2 = 3
    assert d[1] == pytest.approx(-5.0)           # inside the box, 5 from the nearest edge
    assert d[2] == pytest.approx(1.0) and np.allclose(n[2], [0, 1])
    assert d[3] == pytest.approx(3.0)            # box edge at x=20 is nearer than the circle
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_raycast_hits_and_misses():
    obs = Obstacles(circles=[(10, 0, 1)], boxes=[(-5, -20, -3, 20)])
    dist = obs.raycast(np.array([[0.0, 0.0]]), DIRS, 20.0)[0]
    assert dist[0] == pytest.approx(9.0)         # circle ahead
    assert dist[1] == 20.0 and dist[3] == 20.0   # nothing above / below
    assert dist[2] == pytest.approx(3.0)         # box behind


def test_raycast_parallel_ray_outside_box_is_a_miss():
    """Regression: a beam parallel to a box edge but outside it used to 'hit'."""
    obs = Obstacles(boxes=[(95, 0, 105, 53), (95, 67, 105, 120)])
    dist = obs.raycast(np.array([[90.0, 60.0]]), DIRS, 20.0)[0]
    assert dist[0] == 20.0                       # straight through the gap


def test_raycast_matches_dense_sampling():
    rng = np.random.default_rng(3)
    sc = build_scenario("random", np.array([200.0, 120.0]), rng, 25)
    origins = rng.uniform([50, 0], [160, 120], (15, 2))
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    fast = sc.obstacles.raycast(origins, dirs, 20.0)
    ts = np.linspace(0, 20, 2001)
    for a, o in enumerate(origins):
        pts = o[None, None] + ts[None, :, None] * dirs[:, None]          # (R, T, 2)
        sd, _ = sc.obstacles.signed_distance(pts.reshape(-1, 2))
        inside = (sd.reshape(len(dirs), -1) <= 0)
        ref = np.where(inside.any(1), ts[inside.argmax(1)], 20.0)
        assert np.abs(fast[a] - ref).max() < 0.02


def test_resolve_pushes_drone_out():
    obs = Obstacles(circles=[(0, 0, 2)])
    pos, vel = np.array([[1.0, 0.0]]), np.array([[-3.0, 1.0]])
    hit = obs.resolve(pos, vel, radius=0.5)
    assert hit[0]
    assert np.linalg.norm(pos[0]) == pytest.approx(2.5)
    assert vel[0, 0] == pytest.approx(0.0) and vel[0, 1] == pytest.approx(1.0)


# --------------------------------------------------------------- sensor
def test_sensor_detects_nearest_point():
    world = World(100, 100, 10, Obstacles(boxes=[(60, 0, 70, 100)]))
    scan = RangeSensor(36, 20.0).scan(np.array([[50.0, 50.0]]), world)
    det = scan.closest_points()
    assert len(det.i) == 1
    assert det.dist[0] == pytest.approx(10.0)
    assert np.allclose(det.dir[0], [1.0, 0.0])


def test_sensor_sees_nothing_in_open_world():
    world = World(100, 100, 10)
    scan = RangeSensor(16, 20.0).scan(np.array([[50.0, 50.0]]), world)
    assert not scan.hits.any() and len(scan.closest_points().i) == 0


# ------------------------------------------------------------------ APF
def test_apf_pushes_away_from_obstacle():
    cfg = cfg_for()
    world = World(200, 120, 12, Obstacles(circles=[(105, 60, 2)]))
    s = SwarmState(np.array([[100.0, 60.0]]), np.array([[0.0, 0.0]]), np.arange(1))
    scan = RangeSensor(36, 20.0).scan(s.pos, world)
    ctx = StepContext(world, neighbor_pairs(s.pos, 15), None, 0.05, scan, scan.closest_points())
    ctrl = BoidsController(cfg.boids, cfg.drones, cfg.avoidance, use_apf=True)
    ctrl.compute(s, ctx)
    assert ctrl.last_terms.obstacle[0, 0] < 0


# ----------------------------------------------------------------- ORCA
def test_lp_returns_preferred_velocity_when_unconstrained():
    _, v = _lp2([], 8.0, (3.0, 1.0), False, (3.0, 1.0))
    assert v == (3.0, 1.0)


def test_lp_respects_half_plane():
    # constraint: v_x <= 1  (point (1,0), direction (0,1): allowed side is left = x < 1)
    fail, v = _lp2([(1.0, 0.0, 0.0, 1.0)], 8.0, (5.0, 2.0), False, (5.0, 2.0))
    assert fail == 1 and v[0] == pytest.approx(1.0) and v[1] == pytest.approx(2.0)


def test_orca_head_on_drones_do_not_collide():
    from swarm.sim.engine import Simulation
    cfg = cfg_for(drones={"count": 2}, avoidance={"method": "orca"})
    sim = Simulation(cfg)
    sim.state.pos = np.array([[80.0, 60.0], [120.0, 60.0]])
    sim.state.vel = np.array([[5.0, 0.0], [-5.0, 0.0]])
    sim.goal = None
    sim._refresh(0)
    # make both drones want to fly straight at each other
    class Straight:
        def compute(self, state, ctx):
            return np.array([[6.0, 0.0], [-6.0, 0.0]])
    sim.controller = OrcaSafetyLayer(Straight(), cfg.drones, cfg.avoidance)
    seps = [sim.step().min_separation for _ in range(300)]
    assert min(seps) > 2 * cfg.drones.radius


def test_orca_stops_drone_before_wall():
    from swarm.sim.engine import Simulation
    cfg = cfg_for(drones={"count": 1}, avoidance={"method": "orca"})
    sim = Simulation(cfg)
    sim.world.obstacles = Obstacles(boxes=[(120, 0, 130, 120)])
    sim.state.pos = np.array([[90.0, 60.0]])
    sim.state.vel = np.array([[8.0, 0.0]])
    sim.goal = np.array([180.0, 60.0])           # goal is behind the wall
    sim._refresh(0)
    for _ in range(400):
        sim.step()
    assert sim.total_obstacle_hits == 0
    assert sim.metrics.min_clearance > 0


# ------------------------------------------------------------ scenarios
@pytest.mark.parametrize("name", SCENARIOS)
def test_scenarios_build_with_free_spawn(name):
    sc = build_scenario(name, np.array([200.0, 120.0]), np.random.default_rng(0))
    if sc.spawn_region is not None:
        x0, y0, x1, y1 = sc.spawn_region
        grid = np.stack(np.meshgrid(np.linspace(x0, x1, 20), np.linspace(y0, y1, 20)), -1).reshape(-1, 2)
        sd, _ = sc.obstacles.signed_distance(grid)
        assert (sd > 2.0).all()
    if sc.goal is not None:
        sd, _ = sc.obstacles.signed_distance(sc.goal[None])
        assert sd[0] > 5.0


def test_factory_builds_every_method():
    for m in METHODS:
        ctrl = build_controller(cfg_for(avoidance={"method": m}))
        assert isinstance(ctrl, OrcaSafetyLayer) == (m in ("orca", "hybrid"))
    with pytest.raises(ValueError):
        build_controller(cfg_for(avoidance={"method": "teleport"}))


# ------------------------------------------------------------ end to end
@pytest.mark.parametrize("scenario", ["forest", "corridor"])
def test_hybrid_swarm_crosses_safely(scenario):
    from swarm.sim.engine import Simulation
    sim = Simulation(cfg_for(drones={"count": 30}, scenario={"name": scenario},
                             avoidance={"method": "hybrid"}))
    history = sim.run(1600)
    assert sim.total_collisions == 0
    assert sim.total_obstacle_hits == 0
    assert history[-1].arrived >= 0.9 * 30


def test_headless_cli_writes_outputs(tmp_path):
    from swarm.main import main
    main(["--headless", "--scenario", "corridor", "--drones", "10", "--steps", "100",
          "--out", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["scenario"] == "corridor" and summary["avoidance"] == "orca"
    assert (tmp_path / "trajectories.png").exists() and (tmp_path / "metrics.png").exists()


def test_viewer_cycles_methods_and_scenarios(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    from swarm.sim.engine import Simulation
    from swarm.viz.render2d import Viewer
    sim = Simulation(cfg_for(drones={"count": 10}))
    v = Viewer(sim)
    v.show_beams = True
    for key in (pygame.K_2, pygame.K_m, pygame.K_3, pygame.K_m, pygame.K_4):
        v._handle(pygame.event.Event(pygame.KEYDOWN, key=key))
        v._advance(2)
        v._draw_world(); v._draw_swarm(); v._draw_hud()
    assert sim.cfg.scenario.name == "random"
    assert sim.cfg.avoidance.method == "none"      # two presses: orca -> hybrid -> none
    pygame.quit()
