import numpy as np
import pytest

from swarm.agents.state import SwarmState
from swarm.config import load_config
from swarm.control.base import StepContext
from swarm.control.boids import BoidsController
from swarm.mathutil import clamp_norm, norms, safe_normalize
from swarm.sim.dynamics import integrate
from swarm.sim.engine import Simulation
from swarm.sim.neighbors import neighbor_pairs
from swarm.sim.world import World


def make_cfg(**drones):
    return load_config(None, {"drones": drones} if drones else None)


# ----------------------------------------------------------------- maths
def test_clamp_and_normalize():
    v = np.array([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
    c = clamp_norm(v, 1.0)
    assert np.allclose(norms(c), [1.0, 0.5, 0.0])
    u = safe_normalize(v)
    assert np.allclose(norms(u), [1.0, 1.0, 0.0])
    assert not np.isnan(u).any()


# ---------------------------------------------------------------- config
def test_config_defaults_and_overrides():
    cfg = load_config("configs/default.yaml", {"drones": {"count": 7}})
    assert cfg.drones.count == 7
    assert cfg.sim.dt == 0.05


def test_config_rejects_unknown_key():
    with pytest.raises(KeyError):
        load_config(None, {"drones": {"warp_drive": 1}})


# -------------------------------------------------------------- dynamics
def test_integrate_respects_limits():
    world = World(100, 100, 10)
    s = SwarmState(np.full((3, 2), 50.0), np.zeros((3, 2)), np.arange(3))
    huge = np.full((3, 2), 1e6)
    for _ in range(200):
        a = integrate(s, huge, 0.05, max_speed=8.0, max_accel=6.0, world=world, radius=0.5)
        assert (norms(a) <= 6.0 + 1e-9).all()
        assert (norms(s.vel) <= 8.0 + 1e-9).all()
    assert ((s.pos >= 0.5) & (s.pos <= 99.5)).all()


def test_spawn_spacing():
    world = World(200, 120, 12)
    s = SwarmState.spawn(50, world, np.random.default_rng(0), spacing=3.0, speed=0.5)
    nb = neighbor_pairs(s.pos, 3.0)
    assert len(nb.i) == 0


# ----------------------------------------------------------------- boids
def _accel(pos, vel, cfg, goal=None):
    world = World(200, 120, 12)
    s = SwarmState(np.array(pos, float), np.array(vel, float), np.arange(len(pos)))
    ctx = StepContext(world, neighbor_pairs(s.pos, cfg.drones.perception_radius), goal)
    return BoidsController(cfg.boids, cfg.drones).compute(s, ctx)


def test_separation_pushes_apart():
    cfg = make_cfg()
    a = _accel([[100, 60], [101.2, 60]], [[0, 0], [0, 0]], cfg)
    assert a[0, 0] < 0 and a[1, 0] > 0


def test_cohesion_pulls_together():
    cfg = make_cfg()
    cfg.boids.alignment_weight = 0.0
    a = _accel([[90, 60], [102, 60]], [[0, 0], [0, 0]], cfg)
    assert a[0, 0] > 0 and a[1, 0] < 0


def test_alignment_matches_heading():
    cfg = make_cfg()
    cfg.boids.cohesion_weight = 0.0
    a = _accel([[100, 60], [100, 68]], [[0, 5], [5, 0]], cfg)
    # drone 0 heads +y, neighbour heads +x -> drone 0 turns toward +x
    assert a[0, 0] > 0


def test_goal_attracts():
    cfg = make_cfg()
    a = _accel([[50, 60]], [[0, 0]], cfg, goal=np.array([150.0, 60.0]))
    assert a[0, 0] > 0


def test_boundary_pushes_inward():
    cfg = make_cfg()
    a = _accel([[2, 60]], [[-5, 0]], cfg)
    assert a[0, 0] > 0


# ------------------------------------------------------------ simulation
def test_deterministic_with_seed():
    a, b = Simulation(make_cfg(count=20)), Simulation(make_cfg(count=20))
    a.run(200)
    b.run(200)
    assert np.array_equal(a.state.pos, b.state.pos)


@pytest.mark.parametrize("n", [20, 50])
def test_swarm_flies_without_collisions(n):
    sim = Simulation(make_cfg(count=n))
    history = sim.run(1500)
    assert sim.total_collisions == 0
    assert min(m.min_separation for m in history) > 2 * sim.cfg.drones.radius
    assert ((sim.state.pos >= 0) & (sim.state.pos <= sim.world.size)).all()
    late = history[-300:]
    assert np.mean([m.local_polarization for m in late]) > 0.8  # flocks are aligned
    assert np.mean([m.mean_speed for m in late]) > 3.0         # and actually flying


def test_add_remove_drones():
    sim = Simulation(make_cfg(count=10))
    sim.add_drones(5)
    sim.step()
    assert sim.state.n == 15
    sim.remove_drones(100)
    sim.step()
    assert sim.state.n == 1


def test_viewer_smoke(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from swarm.viz.render2d import Viewer
    sim = Simulation(make_cfg(count=15))
    sim.goal = np.array([150.0, 60.0])
    Viewer(sim).run(max_frames=20)
    assert sim.step_count >= 20
