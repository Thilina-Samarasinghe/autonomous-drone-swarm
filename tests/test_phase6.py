"""Phase 6: victims, signals, mission state machine, PSO gossip, auction,
battery, exploration, per-drone planning, rescue end to end."""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from swarm.config import MappingConfig, MissionConfig, load_config
from swarm.mapping.occupancy import OccupancyMaps
from swarm.mission.manager import CHARGE, DEAD, EXPLORE, RETURN, SEEK, RescueMission
from swarm.mission.victims import VictimField
from swarm.sim.scenarios import build_scenario

WORLD = np.array([200.0, 120.0])


def make_mission(n=4, victims=((100.0, 60.0),), **cfg):
    return RescueMission(n, np.array(victims), np.array([20.0, 20.0]), MissionConfig(**cfg),
                         WORLD, 0.05, 30.0, np.random.default_rng(0))


# --------------------------------------------------------------- victims
def test_signal_falls_off_with_distance_and_range():
    f = VictimField(np.array([[0.0, 0.0]]), signal_range=45.0, noise=0.0, rng=np.random.default_rng(0))
    r = f.read(np.array([[1.0, 0.0], [10.0, 0.0], [30.0, 0.0], [50.0, 0.0]]))[:, 0]
    assert r[0] > r[1] > r[2] > 0
    assert r[1] == pytest.approx(0.5)                     # 1 / (1 + 1)
    assert r[3] == 0.0                                     # beyond range


def test_signal_is_noisy():
    f = VictimField(np.array([[0.0, 0.0]]), 45.0, 0.01, np.random.default_rng(1))
    r = f.read(np.repeat([[10.0, 0.0]], 500, axis=0))[:, 0]
    assert 0.005 < r.std() < 0.02 and abs(r.mean() - 0.5) < 0.01


def test_rescue_scenario_layout():
    sc = build_scenario("rescue", WORLD, np.random.default_rng(3), num_victims=8)
    assert len(sc.victims) == 8
    assert (np.linalg.norm(sc.victims - sc.base, axis=1) >= 35).all()
    assert (sc.obstacles.signed_distance(sc.victims)[0] >= 3.0).all()


# ---------------------------------------------------------------- gossip
def test_gossip_merges_found_best_reading_and_bids():
    m = make_mission(n=3, victims=((100, 60), (150, 30)))
    m.found[0, 1] = True
    m.gbest_val[0, 0], m.gbest_pos[0, 0], m.gbest_time[0, 0] = 0.6, [90, 60], 5.0
    m.gbest_val[1, 0], m.gbest_pos[1, 0] = 0.3, [80, 60]
    m.bid[0, 0, 0], m.bid_time[0, 0, 0] = 12.0, 5.0
    m._gossip(np.array([[0, 1], [0, 2]]), np.ones(3, dtype=bool))
    assert m.found[1, 1] and m.found[2, 1]
    assert m.gbest_val[1, 0] == pytest.approx(0.6) and np.allclose(m.gbest_pos[1, 0], [90, 60])
    assert m.bid[2, 0, 0] == 12.0
    # a weaker reading never overwrites a stronger one
    m.gbest_val[0, 0] = 0.1
    m._gossip(np.array([[0, 1]]), np.ones(3, dtype=bool))
    assert m.gbest_val[1, 0] == pytest.approx(0.6)


def test_dead_drones_do_not_gossip():
    m = make_mission(n=2)
    m.found[0, 0] = True
    alive = np.array([False, True])
    m._gossip(np.array([[0, 1]]), alive)
    assert not m.found[1, 0]


# --------------------------------------------------------------- auction
def test_auction_keeps_only_the_closest_seekers():
    m = make_mission(n=5, max_seekers=3)
    t = 10.0
    for j, b in enumerate([5.0, 1.0, 9.0, 3.0, 7.0]):
        m.bid[:, 0, j] = b
        m.bid_time[:, 0, j] = t
    winners = [k for k in range(5) if m._wins(k, 0, t)]
    assert winners == [0, 1, 3]                            # bids 5, 1, 3
    # stale bids don't count
    m.bid_time[:, 0, 1] = t - 100
    assert m._wins(4, 0, t)


# ---------------------------------------------------------------- battery
def test_returns_home_charges_and_resumes():
    m = make_mission(n=1)
    pos = np.array([[120.0, 80.0]])
    vel = np.zeros((1, 2))
    m.battery[0] = m._needed_to_return(pos)[0] - 1e-3     # just below what's needed
    m.update(1, 0.05, pos, vel, None, None)
    assert m.mode[0] == RETURN and np.allclose(m.target[0], m.base)
    m.update(2, 0.10, m.base[None].copy(), vel, None, None)
    assert m.mode[0] == CHARGE
    for s in range(3, 3000):
        m.update(s, s * 0.05, m.base[None].copy(), vel, None, None)
        if m.mode[0] != CHARGE:
            break
    assert m.mode[0] == EXPLORE and m.battery[0] >= 0.98


def test_empty_battery_means_dead():
    m = make_mission(n=1)
    m.battery[0] = 1e-6
    m.update(1, 0.05, np.array([[25.0, 25.0]]), np.array([[5.0, 0.0]]), None, None)
    assert m.mode[0] == DEAD and m.stats.lost == 1


# ------------------------------------------------------------ exploration
def test_frontiers_are_free_cells_next_to_unknown():
    maps = OccupancyMaps(1, WORLD, MappingConfig())
    g = maps.grid(maps.L[0])
    g[20:40, 20:40] = -4.0                                 # a known-free square, unknown around it
    m = make_mission(n=1)
    F = m._frontiers(maps, 0)
    assert len(F) > 0
    # every frontier is on the square's border
    inner = (F[:, 0] > 42) & (F[:, 0] < 78) & (F[:, 1] > 42) & (F[:, 1] < 78)
    assert not inner.any()


def test_voronoi_drone_takes_frontiers_on_its_side():
    maps = OccupancyMaps(2, WORLD, MappingConfig())
    for k in range(2):
        g = maps.grid(maps.L[k])
        g[:, :] = -4.0
        g[30, 15] = g[30, 85] = 0.0                        # two unknown holes: x ~ 30 m and x ~ 170 m
    m = make_mission(n=2, explore="voronoi")

    class Comms:                                           # drone 0 knows drone 1 is far right
        table = np.full((2, 2, 3), np.nan)
    Comms.table[0, 1] = [180.0, 60.0, 0.0]
    m._retarget(np.array([0]), np.array([[100.0, 60.0], [180.0, 60.0]]), maps, Comms, 0.0)
    assert m.target[0, 0] < 100                            # took the left hole, not the nearer-ish right one


# ---------------------------------------------------- per-drone planning
def test_planner_supports_per_drone_goals():
    from swarm.sim.engine import Simulation
    sim = Simulation(load_config(None, {"drones": {"count": 3}, "scenario": {"name": "open"},
                                        "planning": {"enabled": True}}))
    goals = np.array([[150.0, 30.0], [np.nan, np.nan], [40.0, 100.0]])
    targets = sim.planner.update(1, sim.state.pos, goals, sim.maps, 4.0)
    assert np.allclose(sim.planner.paths[0][-1], goals[0], atol=1.5)
    assert np.allclose(sim.planner.paths[2][-1], goals[2], atol=1.5)
    assert sim.planner.paths[1] is None and np.isnan(targets[1]).all()


# --------------------------------------------------- viewer NaN regression
def test_viewer_does_not_hang_on_drone_without_target():
    """pygame loops forever on NaN coordinates; a seeking drone has a NaN waypoint."""
    code = textwrap.dedent("""
        import os, numpy as np
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        from swarm.config import load_config
        from swarm.sim.engine import Simulation
        from swarm.viz.render2d import Viewer
        sim = Simulation(load_config(None, {"drones": {"count": 4}, "scenario": {"name": "rescue"}}))
        v = Viewer(sim)
        sim.run(5)
        sim.planner.carrots[:] = np.nan
        v._draw_world(); v._draw_swarm(); v._draw_hud()
        print("drawn")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "drawn" in out.stdout, out.stderr


# ------------------------------------------------------------ end to end
def rescue_sim(**over):
    from swarm.sim.engine import Simulation
    base = {"drones": {"count": 12}, "scenario": {"name": "rescue"}, "mission": {"victims": 4}}
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return Simulation(load_config(None, base))


def test_full_rescue_finds_and_reports_victims_safely():
    sim = rescue_sim()
    sim.run(5000)                                          # 250 s
    ms = sim.mission.stats
    assert ms.found >= 3 and ms.reported >= 3
    assert ms.lost == 0
    assert sim.total_collisions == 0 and sim.total_obstacle_hits == 0
    seen = set(int(x) for x in np.flatnonzero(sim.mission.stats.mode_steps))
    assert {EXPLORE, SEEK}.issubset(seen)


def test_without_radio_victims_are_only_reported_at_base():
    sim = rescue_sim(comms={"enabled": False}, mission={"victims": 4})
    sim.run(1500)
    m = sim.mission
    # anything the base knows must have been carried home by a drone
    assert m.stats.reported <= m.stats.found


def test_mission_only_runs_with_victims():
    from swarm.sim.engine import Simulation
    assert Simulation(load_config(None, {"drones": {"count": 3}})).mission is None
    assert rescue_sim(mission={"enabled": False}).mission is None


def test_cli_rescue_flags(tmp_path):
    from swarm.main import main
    main(["--headless", "--scenario", "rescue", "--drones", "6", "--steps", "300", "--victims", "3",
          "--explore", "frontier", "--seek", "none", "--out", str(tmp_path)])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["explore"] == "frontier" and s["seek"] == "none" and s["victims"] == 3
    assert len(s["victim_found_times_s"]) == 3 and s["drones_lost"] == 0
