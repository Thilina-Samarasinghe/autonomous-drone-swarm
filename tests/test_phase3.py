"""Phase 3: occupancy mapping, radio network, map sharing."""

import json

import numpy as np
import pytest

from swarm.agents.sensors import RangeSensor
from swarm.comms.network import CommsNetwork
from swarm.config import CommsConfig, MappingConfig, load_config
from swarm.mapping.occupancy import GroundTruth, OccupancyMaps
from swarm.sim.obstacles import Obstacles
from swarm.sim.world import World

SIZE = np.array([100.0, 60.0])


def make_maps(n=1):
    return OccupancyMaps(n, SIZE, MappingConfig())


# ------------------------------------------------------------- mapping
def test_scan_marks_free_space_and_wall():
    world = World(100, 60, 10, Obstacles(boxes=[(60, 0, 70, 60)]))
    maps = make_maps()
    pos = np.array([[45.0, 30.0]])                         # wall 15 m away (lidar range 20 m)
    scan = RangeSensor(72, 20.0).scan(pos, world)
    for _ in range(5):
        maps.integrate_scan(pos, scan)
    free_cell, _ = maps.cell_of(np.array([50.0, 30.0]))
    wall_cell, _ = maps.cell_of(np.array([60.5, 30.0]))
    unseen_cell, _ = maps.cell_of(np.array([90.0, 30.0]))   # behind the wall
    assert maps.L[0, free_cell] < -1.0
    assert maps.L[0, wall_cell] > 1.0
    assert maps.L[0, unseen_cell] == 0.0


def test_log_odds_are_clamped():
    world = World(100, 60, 10, Obstacles(boxes=[(60, 0, 70, 60)]))
    maps = make_maps()
    pos = np.array([[40.0, 30.0]])
    scan = RangeSensor(36, 20.0).scan(pos, world)
    for _ in range(50):
        maps.integrate_scan(pos, scan)
    assert np.abs(maps.L).max() <= MappingConfig().l_max + 1e-6


def test_mapping_accuracy_against_ground_truth():
    obs = Obstacles(circles=[(30, 30, 3), (70, 20, 4)], boxes=[(45, 40, 55, 50)])
    world = World(100, 60, 10, obs)
    maps = make_maps()
    truth = GroundTruth(maps, obs)
    sensor = RangeSensor(72, 20.0)
    for x in np.linspace(10, 90, 30):                     # fly a lawn-mower sweep
        for y in (10.0, 30.0, 50.0):
            p = np.array([[x, y]])
            if obs.signed_distance(p)[0][0] > 1:
                maps.integrate_scan(p, sensor.scan(p, world))
    assert truth.coverage(maps.L, 1.0)[0] > 0.6
    assert truth.accuracy(maps.L, 1.0) > 0.97


# ------------------------------------------------------------- merging
def test_merge_more_confident_wins_and_is_idempotent():
    maps = make_maps(2)
    maps.L[0, [1, 2, 3]] = [3.5, -3.5, 0.5]
    maps.L[1, [1, 2, 3]] = [1.0, -3.4, 2.0]
    idx = np.array([1, 2, 3])
    adopted = maps.merge(1, idx, maps.L[0, idx].copy(), relay=True)
    assert adopted == 1                                    # only cell 1 (3.5 vs 1.0)
    assert maps.L[1, 1] == pytest.approx(3.5)
    assert maps.L[1, 2] == pytest.approx(-3.4)             # within margin: keep own
    assert maps.L[1, 3] == pytest.approx(2.0)              # own is more confident
    assert maps.dirty[1, 1] and not maps.dirty[1, 3]
    assert maps.merge(1, idx, maps.L[0, idx].copy(), relay=True) == 0   # idempotent


def test_merge_duplicates_take_most_confident():
    maps = make_maps(1)
    adopted = maps.merge(0, np.array([5, 5, 5]), np.array([1.0, -3.9, 2.0], dtype=np.float32), relay=False)
    assert adopted == 1 and maps.L[0, 5] == pytest.approx(-3.9)
    assert not maps.dirty[0, 5]                            # relay off: not re-broadcast


def test_take_message_sends_dirty_then_refresh():
    maps = make_maps(1)
    rng = np.random.default_rng(0)
    maps.L[0, :50] = 3.0
    maps.dirty[0, :10] = True
    cells, vals = maps.take_message(0, max_cells=100, rng=rng)
    assert set(range(10)) <= set(cells.tolist())           # dirty cells first
    assert len(cells) > 10                                 # plus anti-entropy refresh
    assert not maps.dirty[0].any()
    assert np.allclose(vals, 3.0)


def test_take_message_own_only_skips_relayed_cells():
    maps = make_maps(1)
    maps.L[0, :20] = 3.0
    maps.own[0, :5] = True                                 # only 5 cells seen first-hand
    cells, _ = maps.take_message(0, max_cells=100, rng=np.random.default_rng(0), own_only=True)
    assert set(cells.tolist()) <= set(range(5))


# --------------------------------------------------------------- radio
def make_net(n, **cfg):
    c = CommsConfig(**cfg)
    return CommsNetwork(n, c, dt=0.05, rng=np.random.default_rng(1))


def run_net(net, pos, steps, maps=None):
    for k in range(1, steps + 1):
        net.step(k, k * 0.05, pos, maps)


def test_drones_in_range_hear_each_other():
    net = make_net(2, range=30.0, latency=0.0)
    run_net(net, np.array([[0.0, 0.0], [20.0, 0.0]]), 10)
    assert not np.isnan(net.table[0, 1, 2]) and not np.isnan(net.table[1, 0, 2])
    assert net.stats.dropped == 0 and net.stats.delivered > 0


def test_out_of_range_never_hear():
    net = make_net(2, range=30.0)
    run_net(net, np.array([[0.0, 0.0], [50.0, 0.0]]), 20)
    assert np.isnan(net.table[0, 1, 2]) and net.stats.attempted == 0
    assert net.groups() == 2


def test_total_packet_loss_delivers_nothing():
    net = make_net(2, packet_loss=1.0)
    run_net(net, np.array([[0.0, 0.0], [10.0, 0.0]]), 20)
    assert net.stats.delivered == 0 and net.stats.dropped == net.stats.attempted > 0
    assert np.isnan(net.table[0, 1, 2])


def test_latency_delays_delivery():
    net = make_net(2, latency=0.5, broadcast_interval=0.05)   # 10 steps of latency
    pos = np.array([[0.0, 0.0], [10.0, 0.0]])
    run_net(net, pos, 9)
    assert np.isnan(net.table[0, 1, 2])
    run_net_from = 10
    net.step(run_net_from, run_net_from * 0.05, pos, None)
    net.step(run_net_from + 1, (run_net_from + 1) * 0.05, pos, None)
    assert not np.isnan(net.table[0, 1, 2])


@pytest.mark.parametrize("relay", [True, False])
def test_multi_hop_relay(relay):
    # A -- B -- C in a line; A and C are out of each other's range
    pos = np.array([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0]])
    net = make_net(3, range=30.0, relay=relay, latency=0.0)
    run_net(net, pos, 40)
    assert np.isnan(net.table[0, 2, 2]) == (not relay)      # A knows C only via relay
    assert not np.isnan(net.table[0, 1, 2])                  # A always knows B


@pytest.mark.parametrize("relay", [True, False])
def test_map_cells_travel_multi_hop_only_with_relay(relay):
    pos = np.array([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0]])
    maps = make_maps(3)
    maps.L[2, 100] = 3.5                                     # only C knows cell 100
    maps.own[2, 100] = True
    maps.dirty[2, 100] = True
    net = make_net(3, range=30.0, relay=relay, latency=0.0)
    run_net(net, pos, 40, maps)
    assert maps.L[1, 100] == pytest.approx(3.5)              # B always hears C
    assert (maps.L[0, 100] == pytest.approx(3.5)) == relay   # A only via relay


# --------------------------------------------------------- end to end
def test_sharing_makes_every_drone_know_more():
    from swarm.sim.engine import Simulation
    cov = {}
    for share in (False, True):
        sim = Simulation(load_config(None, {"drones": {"count": 20}, "scenario": {"name": "survey"},
                                            "comms": {"share_maps": share}}))
        sim.run(600)                                         # 30 s
        im = sim.info_metrics()
        cov[share] = im.map_coverage_mean
        assert im.map_accuracy > 0.98
        assert sim.total_collisions == 0 and sim.total_obstacle_hits == 0
    assert cov[True] > cov[False] + 0.3


def test_radio_randomness_does_not_change_flight():
    from swarm.sim.engine import Simulation
    runs = []
    for loss in (0.0, 0.5):
        sim = Simulation(load_config(None, {"drones": {"count": 15}, "scenario": {"name": "survey"},
                                            "comms": {"packet_loss": loss}}))
        sim.run(200)
        runs.append(sim.state.pos.copy())
    assert np.array_equal(runs[0], runs[1])


def test_add_drones_resizes_maps_and_radio():
    from swarm.sim.engine import Simulation
    sim = Simulation(load_config(None, {"drones": {"count": 10}}))
    sim.run(20)
    sim.add_drones(5)
    sim.run(20)
    assert sim.maps.n == 15 and sim.comms.n == 15
    sim.remove_drones(3)
    sim.run(20)
    assert sim.maps.n == 12 and sim.comms.n == 12


def test_bool_config_parses_strings():
    assert load_config(None, {"comms": {"share_maps": "false"}}).comms.share_maps is False
    assert load_config(None, {"comms": {"relay": "yes"}}).comms.relay is True
    with pytest.raises(ValueError):
        load_config(None, {"comms": {"relay": "maybe"}})


def test_cli_comms_flags(tmp_path):
    from swarm.main import main
    main(["--headless", "--scenario", "survey", "--drones", "10", "--steps", "100",
          "--packet-loss", "0.3", "--no-relay", "--out", str(tmp_path)])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["packet_loss"] == 0.3 and s["relay"] is False and s["map_sharing"] is True
    assert 0 <= s["map_coverage_mean_final"] <= 1
    assert (tmp_path / "comms_mapping.png").exists()


def test_viewer_comms_keys(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    from swarm.sim.engine import Simulation
    from swarm.viz.render2d import Viewer
    sim = Simulation(load_config(None, {"drones": {"count": 12}, "scenario": {"name": "survey"}}))
    v = Viewer(sim)
    for key in (pygame.K_g, pygame.K_TAB, pygame.K_n, pygame.K_p, pygame.K_l, pygame.K_g, pygame.K_5):
        v._handle(pygame.event.Event(pygame.KEYDOWN, key=key))
        v._advance(3)
        v._draw_world(); v._draw_swarm(); v._draw_hud()
    assert v.map_mode == "team" and v.selected == 1 and v.link_mode == "sensing"
    assert sim.cfg.comms.share_maps is False and sim.cfg.comms.packet_loss == pytest.approx(0.2)
    pygame.quit()
