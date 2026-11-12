"""Phase 9: fault injection (drone failures, leader re-election, degraded lidar), demo, experiments."""

import numpy as np
import pytest

from swarm.config import load_config
from swarm.sim.engine import Simulation


def sim_with(**over):
    return Simulation(load_config(None, over))


# ------------------------------------------------------------ drone failures
def test_failed_drones_land_and_disappear_from_the_swarm():
    sim = sim_with(drones={"count": 10}, scenario={"name": "survey"})
    sim.run(20)
    sim.fail([2, 5])
    where = sim.state.pos[[2, 5]].copy()
    sim.run(100)
    assert not sim.alive[[2, 5]].any() and sim.alive.sum() == 8
    assert np.allclose(sim.state.pos[[2, 5]], where)                     # stayed where they landed
    assert np.allclose(sim.state.vel[[2, 5]], 0.0)
    nb = sim.neighbors
    assert not np.isin(nb.i, [2, 5]).any() and not np.isin(nb.j, [2, 5]).any()
    assert sim.metrics.n == 8                                            # metrics: flying drones only
    links = sim.comms.links
    assert not np.isin(links, [2, 5]).any()                              # radios silent


def test_failed_drone_maps_stop_changing():
    sim = sim_with(drones={"count": 6}, scenario={"name": "survey"}, comms={"enabled": False})
    sim.run(20)
    sim.fail([0])
    before = sim.maps.L[0].copy()
    sim.run(100)
    assert np.array_equal(sim.maps.L[0], before)
    assert not np.array_equal(sim.maps.L[1], before)


def test_scheduled_failures_hit_the_right_number_of_drones():
    sim = sim_with(drones={"count": 20}, faults={"fail_time": 1.0, "fail_fraction": 0.25})
    sim.run(10)
    assert sim.alive.all()
    sim.run(20)
    assert int((~sim.alive).sum()) == 5 and sim.alive[0]                 # leader spared unless asked
    assert sim.failed_at == pytest.approx(1.0, abs=0.06)


def test_formation_re_elects_a_leader_and_reforms():
    sim = sim_with(drones={"count": 12}, formation={"enabled": True, "shape": "v"},
                   faults={"fail_time": 15.0, "fail_fraction": 0.25, "fail_leader": True})
    sim.run(int(16 / 0.05))
    f = sim.formation
    assert not sim.alive[0]
    assert f.leader == int(np.flatnonzero(sim.alive)[0]) and f.reelections == 1
    assert len(f.offsets) == int(sim.alive.sum())                       # shape resized for survivors
    sim.run(int(40 / 0.05))
    assert f.stats.last_error < 2.5                                     # back in shape
    assert sim.total_collisions == 0


def test_rescue_marks_failed_drones_dead():
    from swarm.mission.manager import DEAD
    sim = sim_with(drones={"count": 8}, scenario={"name": "rescue"}, mission={"victims": 3})
    sim.run(40)
    sim.fail([3])
    sim.run(40)
    assert sim.mission.mode[3] == DEAD


def test_radio_groups_ignore_failed_drones():
    sim = sim_with(drones={"count": 10})
    sim.run(10)
    groups_before = sim.comms.groups()
    sim.fail([1, 2, 3])
    sim.run(10)
    assert sim.comms.groups() <= groups_before


# --------------------------------------------------------------- lidar faults
def test_lidar_noise_and_dropout_degrade_the_scan():
    clean = sim_with(drones={"count": 6}, scenario={"name": "forest"})
    noisy = sim_with(drones={"count": 6}, scenario={"name": "forest"}, faults={"lidar_noise": 1.0})
    dropped = sim_with(drones={"count": 6}, scenario={"name": "forest"}, faults={"lidar_dropout": 1.0})
    for s in (clean, noisy, dropped):
        s.state.pos[0] = [62.0, 60.0]                                    # next to a pillar
        s._refresh(0)
    hit = clean.scan.dist[0] < clean.scan.max_range - 1e-9
    assert hit.any()
    assert not np.allclose(noisy.scan.dist[0][hit], clean.scan.dist[0][hit])
    assert np.allclose(dropped.scan.dist, dropped.scan.max_range)       # every beam lost


def test_no_faults_configured_changes_nothing():
    a = sim_with(drones={"count": 8}, scenario={"name": "forest"})
    b = sim_with(drones={"count": 8}, scenario={"name": "forest"}, faults={"lidar_noise": 0.0})
    a.run(100)
    b.run(100)
    assert np.array_equal(a.state.pos, b.state.pos)


def test_viewer_draws_failed_drones(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    from swarm.viz.render2d import Viewer
    sim = sim_with(drones={"count": 8}, formation={"enabled": True, "shape": "v"})
    v = Viewer(sim)
    sim.fail([0, 3])
    v._advance(2)
    v._draw_world(); v._draw_swarm(); v._draw_hud()
    pygame.quit()


# ------------------------------------------------------- demo and experiments
def test_demo_gif_is_written(tmp_path, monkeypatch):
    from experiments import make_demo
    monkeypatch.setitem(make_demo.DEMOS, "test_tiny", ("tiny", {"drones": {"count": 4}}, 1.0, 5, {}))
    path = make_demo.record("test_tiny", tmp_path, scale=0.3, fps=10)
    from PIL import Image
    im = Image.open(path)
    assert im.n_frames == 4 and im.size == (360, 228)


def test_robustness_case_runs():
    from experiments import robustness
    case = next(c for c in robustness.cases() if c["suite"] == "failures" and c["kind"] == "nav")
    case = dict(case, drones=8, steps=300)
    row = robustness.run_one((case, 1))
    assert row["alive"] < 8 and row["collision_steps"] == 0 and "arrived_pct" in row


def test_cli_fault_flags(tmp_path):
    import json
    from swarm.main import main
    main(["--headless", "--formation", "v", "--drones", "8", "--steps", "120", "--fail-time", "2",
          "--fail-fraction", "0.25", "--fail-leader", "--lidar-noise", "0.2", "--out", str(tmp_path)])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["drones_failed"] == 3                                     # 2 random + the leader
