"""Phase 5: formation shapes, assignment, leader trail, squeezing, radio-based following."""

import json
import math

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from swarm.comms.network import CommsNetwork
from swarm.config import CommsConfig, FormationConfig, MappingConfig, load_config
from swarm.formation.manager import FormationManager
from swarm.formation.shapes import SHAPES, slot_offsets, to_world, width
from swarm.mapping.occupancy import OccupancyMaps


# ---------------------------------------------------------------- shapes
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n", [1, 2, 9, 20, 50])
def test_shapes_have_n_slots_leader_first_and_spacing(shape, n):
    o = slot_offsets(shape, n, 5.0)
    assert o.shape == (n, 2)
    assert np.allclose(o[0], 0.0)
    if n > 1:
        assert pdist(o).min() >= 5.0 * 0.99


def test_shape_widths_and_unknown_shape():
    assert width(slot_offsets("column", 20, 5.0)) == 0.0
    assert width(slot_offsets("line", 5, 5.0)) == pytest.approx(20.0)
    assert width(slot_offsets("v", 20, 5.0)) > 14.0          # wider than the corridor gap
    with pytest.raises(ValueError):
        slot_offsets("star", 5, 5.0)


def test_to_world_rotates_with_heading():
    o = np.array([[0.0, 0.0], [-5.0, 0.0], [0.0, 5.0]])      # behind, left
    w = to_world(o, np.array([10.0, 10.0]), math.pi / 2)     # heading north
    assert np.allclose(w, [[10, 10], [10, 5], [5, 10]])


# ------------------------------------------------------------ assignment
def test_hungarian_assignment_is_optimal():
    n = 8
    m = FormationManager(n, FormationConfig(shape="box"), 0.05, np.array([200.0, 120.0]))
    m.heading = 0.0
    origin = np.array([100.0, 60.0])
    slots = to_world(m.offsets, origin, 0.0)
    rng = np.random.default_rng(0)
    perm = np.r_[0, 1 + rng.permutation(n - 1)]               # drones sit exactly on shuffled slots
    pos = slots[perm]
    m._assign(pos, origin)
    assert np.allclose(slots[m.assign], pos)                 # every drone got the slot it sits on


# ----------------------------------------------------------------- trail
def make_manager(n=5, **cfg):
    m = FormationManager(n, FormationConfig(**cfg), 0.05, np.array([200.0, 120.0]))
    m.heading = 0.0
    return m


def test_trail_points_follow_the_flown_path_around_a_corner():
    m = make_manager()
    # leader flew east 20 m, then north 20 m (an L)
    for x in np.arange(0, 20.5, 0.5):
        m._record_trail(np.array([x, 0.0]))
    for y in np.arange(0.5, 20.5, 0.5):
        m._record_trail(np.array([20.0, y]))
    pts, tang = m._trail_points(np.array([5.0, 25.0, 30.0]))
    assert np.allclose(pts[0], [20, 15], atol=1e-6)          # 5 m back: still on the north leg
    assert np.allclose(pts[1], [15, 0], atol=1e-6)           # 25 m back: around the corner
    assert np.allclose(tang[1], [1, 0], atol=1e-6)           # travel direction there: east
    assert np.allclose(pts[2], [10, 0], atol=1e-6)


def test_trail_extends_straight_back_beyond_its_start():
    m = make_manager()
    for x in np.arange(0, 5.5, 1.0):
        m._record_trail(np.array([x, 0.0]))
    pts, _ = m._trail_points(np.array([10.0]))
    assert np.allclose(pts[0], [-5.0, 0.0], atol=1e-6)


def test_follower_errors_measure_slot_or_trail_distance():
    m = make_manager(n=3, shape="line")
    pos = to_world(m.offsets, np.array([50.0, 50.0]), 0.0)      # everyone exactly on its slot
    m._assign(pos, pos[0])
    assert np.allclose(m.follower_errors(pos), 0.0, atol=1e-9)
    pos[2] += [0.0, 7.0]
    assert np.allclose(sorted(m.follower_errors(pos)), [0.0, 7.0])
    # column: errors are measured against points on the leader's trail
    for x in np.arange(0, 30.5, 1.0):
        m._record_trail(np.array([x, 50.0]))
    m._activate("column")
    m.assign = np.array([0, 1, 2])
    col = np.array([[30.0, 50.0], [25.0, 50.0], [20.0, 53.0]])
    assert np.allclose(m.follower_errors(col), [0.0, 3.0], atol=1e-6)


# ------------------------------------------------------- passage width
def test_passage_width_measured_on_leaders_map():
    maps = OccupancyMaps(1, np.array([100.0, 60.0]), MappingConfig())
    g = maps.grid(maps.L[0])
    g[:, :] = -4.0
    g[0:12, 20:40] = 4.0                                       # walls below y=24 and above y=36
    g[18:, 20:40] = 4.0                                        # -> a 12 m corridor at x 40..80
    m = make_manager(n=1)
    w = m._passage_width(np.array([30.0, 30.0]), None, maps)   # heading east into the corridor
    assert 10.0 <= w <= 14.0
    assert m._passage_width(np.array([30.0, 30.0]), None, None) == float("inf")


def test_segments_clear_detects_walls():
    maps = OccupancyMaps(1, np.array([40.0, 40.0]), MappingConfig())
    maps.grid(maps.L[0])[:, 10] = 4.0                          # wall at x = 20..22
    k = np.array([0, 0])
    p = np.array([[5.0, 5.0], [5.0, 5.0]])
    q = np.array([[15.0, 30.0], [35.0, 5.0]])
    assert maps.segments_clear(k, p, q).tolist() == [True, False]


# ---------------------------------------------------- radio velocities
def test_beacons_carry_velocity():
    net = CommsNetwork(2, CommsConfig(latency=0.0), 0.05, np.random.default_rng(0))
    pos = np.array([[0.0, 0.0], [10.0, 0.0]])
    vel = np.array([[3.0, 1.0], [0.0, 0.0]])
    for k in range(1, 8):
        net.step(k, k * 0.05, pos, None, vel)
    assert np.allclose(net.vtable[1, 0], [3.0, 1.0])


# ------------------------------------------------------------ end to end
def sim_with(**over):
    from swarm.sim.engine import Simulation
    return Simulation(load_config(None, over))


def test_v_formation_holds_shape_in_open_flight():
    sim = sim_with(drones={"count": 12}, formation={"enabled": True, "shape": "v"})
    sim.run(1600)
    fs = sim.formation.stats
    assert fs.in_shape_fraction > 0.6
    assert sim.total_collisions == 0


def test_formation_squeezes_through_gap_and_reforms():
    sim = sim_with(drones={"count": 12}, scenario={"name": "corridor"},
                   formation={"enabled": True, "shape": "v"})
    shapes = set()
    for _ in range(2600):
        sim.step()
        shapes.add(sim.formation.active_shape)
    assert "column" in shapes and sim.formation.stats.squeezes >= 1
    assert sim.formation.active_shape == "v"                 # restored after the gap
    assert sim.time_leader_arrived is not None
    assert sim.total_collisions == 0 and sim.total_obstacle_hits == 0
    far = np.linalg.norm(sim.state.pos[1:] - sim.state.pos[0], axis=1) > 50
    assert not far.any()                                     # nobody left behind


def test_without_squeeze_the_trail_fallback_still_gets_everyone_through():
    """Squeezing off: the line keeps its shape, followers whose slot is behind
    the wall follow the leader's trail through the gap instead."""
    sim = sim_with(drones={"count": 12}, scenario={"name": "corridor"},
                   formation={"enabled": True, "shape": "line", "adaptive": False})
    used_trail = False
    for _ in range(2600):
        sim.step()
        used_trail |= bool(sim.formation.following_trail.any())
    assert sim.formation.stats.squeezes == 0
    assert used_trail
    assert sim.time_leader_arrived is not None
    far = np.linalg.norm(sim.state.pos[1:] - sim.state.pos[0], axis=1) > 50
    assert not far.any()
    assert sim.total_collisions == 0 and sim.total_obstacle_hits == 0


def test_followers_without_leader_info_fall_back_to_flocking():
    sim = sim_with(drones={"count": 6}, scenario={"name": "open"}, comms={"packet_loss": 1.0},
                   formation={"enabled": True, "shape": "line"})
    sim.state.pos[5] = [180.0, 100.0]                         # far beyond sensing range of the leader
    sim._init_info_systems()
    sim._refresh(0)
    for _ in range(int(3.5 / sim.cfg.sim.dt)):                # longer than leader_timeout
        sim.step()
    ft = sim.last_formation_targets
    if np.linalg.norm(sim.state.pos[5] - sim.state.pos[0]) > sim.cfg.drones.perception_radius:
        assert np.isnan(ft.slot[5, 0])


def test_set_formation_switches_and_disables():
    sim = sim_with(drones={"count": 10})
    sim.set_formation("circle")
    sim.run(20)
    assert sim.formation.active_shape == "circle"
    sim.set_formation("box")
    sim.run(20)
    assert sim.formation.active_shape == "box" and sim.formation.stats.switches >= 2
    sim.set_formation(None)
    sim.run(20)
    assert sim.formation is None and sim.last_formation_targets is None


def test_unknown_shape_rejected_by_simulation():
    with pytest.raises(ValueError):
        sim_with(drones={"count": 3}, formation={"enabled": True, "shape": "star"})


def test_cli_formation_flags(tmp_path):
    from swarm.main import main
    main(["--headless", "--scenario", "corridor", "--drones", "8", "--steps", "300",
          "--formation", "v", "--out", str(tmp_path)])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["formation"] == "v" and "formation_error_mean_m" in s and "leader_arrived_s" in s


def test_viewer_formation_key(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    from swarm.viz.render2d import Viewer
    sim = sim_with(drones={"count": 8})
    v = Viewer(sim)
    seen = []
    for _ in range(len(SHAPES) + 1):
        v._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        seen.append(sim.formation.desired_shape if sim.formation else None)
        v._advance(3)
        v._draw_world(); v._draw_swarm(); v._draw_hud()
    assert seen == list(SHAPES) + [None]
    pygame.quit()
