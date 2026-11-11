"""Phase 8: observations, training environments, PPO, deploying learned policies."""

import json

import numpy as np
import pytest
import torch

from swarm.config import MappingConfig, load_config
from swarm.mapping.occupancy import OccupancyMaps
from swarm.rl.env import TASKS, SwarmEnv
from swarm.rl.observe import OBS_DIM, Observer
from swarm.rl.ppo import Actor, PPOConfig, load_policy, save_policy, train

torch.set_num_threads(1)


# ---------------------------------------------------------- observations
def test_observation_layout():
    ob = Observer(max_speed=8.0, perception=15.0)
    pos = np.array([[50.0, 50.0], [53.0, 50.0], [100.0, 100.0]])
    vel = np.array([[4.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    targets = np.array([[90.0, 50.0], [np.nan, np.nan], [101.0, 100.0]])
    o = ob(pos, vel, targets)
    assert o.shape == (3, OBS_DIM) and np.isfinite(o).all()
    assert np.allclose(o[0, 0:2], [0.5, 0.0])
    assert np.allclose(o[0, 2:4], [1.0, 0.0])            # 40 m away: clipped to length 1
    assert np.allclose(o[1, 2:5], 0.0)                   # no target
    assert o[2, 4] == pytest.approx(np.tanh(1 / 20))
    assert np.allclose(o[0, 5:10], [3 / 15, 0, -0.5, 0, 1])   # nearest neighbour first
    assert np.allclose(o[2, 5:10], 0.0)                  # nobody within 15 m
    assert np.allclose(o[:, 35:47], 1.0)                 # no lidar: free everywhere


def test_geofence_shows_up_in_the_lidar_sectors():
    """The arena edge must be visible, or the policy is punished for walls it can't see."""
    ob = Observer(8.0, 15.0, world_size=np.array([100.0, 100.0]))
    o = ob(np.array([[95.0, 50.0], [50.0, 50.0]]), np.zeros((2, 2)), None)
    east = o[0, 35]                                      # first sector points (almost) east
    assert east < 0.3                                    # wall ~5 m away
    assert np.allclose(o[1, 35:47], 1.0)                 # in the middle: nothing within 20 m


def test_velocity_commands_become_bounded_accelerations():
    from swarm.rl.observe import action_to_accel
    acc = action_to_accel(np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[8.0, 0.0], [4.0, 0.0]]), 8.0, 6.0)
    assert np.allclose(acc[0], 0.0)                      # already at the commanded velocity
    assert np.allclose(acc[1], [-6.0, 0.0])              # brake, capped at max_accel


def test_map_sectors_report_unknown_space():
    maps = OccupancyMaps(1, np.array([100.0, 100.0]), MappingConfig())
    ob = Observer(8.0, 15.0, maps)
    pos, vel = np.array([[50.0, 50.0]]), np.zeros((1, 2))
    assert np.allclose(ob(pos, vel, None, maps=maps)[0, 47:55], 1.0)      # all unknown
    maps.L[:] = -4.0
    assert np.allclose(ob(pos, vel, None, maps=maps)[0, 47:55], 0.0)      # all known


# ------------------------------------------------------------ environments
@pytest.mark.parametrize("task", TASKS)
def test_environments_step(task):
    env = SwarmEnv(task, n=4, seed=0)
    obs = env.reset()
    assert obs.shape == (4, OBS_DIM)
    for _ in range(5):
        obs, r, done, info = env.step(np.zeros((4, 2)))
        assert r.shape == (4,) and np.isfinite(r).all() and np.isfinite(obs).all()
    assert env.global_state(obs).shape == (4, 2 * OBS_DIM)


def test_navigate_rewards_progress_toward_the_goal():
    env = SwarmEnv("navigate", n=1, seed=3)
    env.reset()
    to = env.goals[0] - env.sim.state.pos[0]
    toward = (to / np.linalg.norm(to))[None]
    rs_toward = [env.step(toward)[1][0] for _ in range(20)]
    env.reset()
    to = env.goals[0] - env.sim.state.pos[0]
    away = -(to / np.linalg.norm(to))[None]
    rs_away = [env.step(away)[1][0] for _ in range(20)]
    assert sum(rs_toward) > sum(rs_away)


def test_formation_leader_is_scripted_and_not_trained():
    env = SwarmEnv("formation", n=5, seed=0)
    assert env.trainable.tolist() == [False, True, True, True, True]
    env.reset()
    start = env.sim.state.pos[0].copy()
    for _ in range(40):
        env.step(np.zeros((5, 2)))                       # zero action for everyone ...
    assert np.linalg.norm(env.sim.state.pos[0] - start) > 2.0   # ... but the leader still flies


def test_collisions_are_penalised():
    env = SwarmEnv("flock", n=2, seed=0)
    env.reset()
    env.sim.state.pos[:] = [[100.0, 60.0], [100.6, 60.0]]          # overlapping
    env.sim.state.vel[:] = 0.0
    env.sim._refresh(0)
    _, r, _, info = env.step(np.zeros((2, 2)))
    assert info["collisions"] == 2 and (r < -2.0).all()


# --------------------------------------------------------------------- PPO
def test_ppo_runs_and_policy_roundtrips(tmp_path):
    env = SwarmEnv("flock", n=4, seed=0)
    actor, log = train(env, 2 * 256 * 4, PPOConfig(rollout_steps=256), log_every=0)
    assert len(log.updates) == 2 and np.isfinite(log.updates[-1]["return"])
    path = tmp_path / "p.pt"
    save_policy(actor, str(path), {"task": "flock"})
    loaded, meta = load_policy(str(path))
    obs = np.random.default_rng(0).normal(size=(5, OBS_DIM)).astype(np.float32)
    assert meta["task"] == "flock"
    assert np.allclose(actor.act(obs), loaded.act(obs))
    assert (np.abs(loaded.act(obs)) <= 1.0).all()


class _ToyGoalEnv:
    """Pure-maths sanity task: each agent sees the direction to the origin and
    is rewarded for moving toward it (no simulator)."""
    obs_dim, n = OBS_DIM, 4
    trainable = np.ones(4, dtype=bool)

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self):
        self.p, self.t = self.rng.uniform(-10, 10, (4, 2)), 0
        return self._obs()

    def _obs(self):
        o = np.zeros((4, OBS_DIM), dtype=np.float32)
        o[:, 2:4] = -self.p / 10
        return o

    def global_state(self, o):
        return np.concatenate([o, np.repeat(o.mean(0, keepdims=True), 4, 0)], 1)

    def step(self, a):
        d0 = np.linalg.norm(self.p, axis=1)
        self.p = self.p + np.clip(a, -1, 1) * 0.5
        self.t += 1
        return self._obs(), (d0 - np.linalg.norm(self.p, axis=1)).astype(np.float32), self.t >= 50, {}


def test_ppo_learns_a_trivial_task():
    """PPO itself is correct: on a trivial task the return clearly improves."""
    actor, log = train(_ToyGoalEnv(), 40 * 256 * 4, PPOConfig(rollout_steps=256, minibatch=512), log_every=0)
    rets = [u["return"] for u in log.updates if np.isfinite(u["return"])]
    assert np.mean(rets[-5:]) > np.mean(rets[:5]) + 1.0


# ------------------------------------------------------------- deployment
@pytest.fixture(scope="module")
def untrained_policies(tmp_path_factory):
    d = tmp_path_factory.mktemp("pol")
    paths = {}
    for task in TASKS:
        p = d / f"{task}.pt"
        save_policy(Actor(), str(p), {"task": task})
        paths[task] = str(p)
    return paths


@pytest.mark.parametrize("task,scenario,extra", [
    ("navigate", "forest", {}), ("flock", "open", {}),
    ("formation", "open", {"formation": {"enabled": True, "shape": "v"}}), ("search", "survey", {})])
def test_policies_fly_in_the_simulator(untrained_policies, task, scenario, extra):
    from swarm.sim.engine import Simulation
    over = {"drones": {"count": 6}, "scenario": {"name": scenario}, "rl": {"policy": untrained_policies[task]}}
    over.update(extra)
    sim = Simulation(load_config(None, over))
    sim.run(40)
    assert sim.controller.task == task
    assert np.isfinite(sim.state.pos).all()


def test_policy_with_orca_never_collides(untrained_policies):
    from swarm.sim.engine import Simulation
    sim = Simulation(load_config(None, {"drones": {"count": 10}, "scenario": {"name": "forest"},
                                        "rl": {"policy": untrained_policies["navigate"], "orca": True}}))
    sim.run(400)
    assert sim.total_collisions == 0 and sim.total_obstacle_hits == 0


def test_cli_policy_flag(untrained_policies, tmp_path):
    from swarm.main import main
    main(["--headless", "--scenario", "forest", "--drones", "5", "--steps", "60",
          "--policy", untrained_policies["navigate"], "--rl-orca", "--out", str(tmp_path)])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["policy"].endswith("navigate.pt") and s["policy_with_orca"] is True
