"""Fly a trained policy inside the normal simulator (implements Controller).

The same Observer builds each drone's local observation as in training; the
shared actor turns it into an acceleration. Targets per task:
  navigate   planner waypoint if planning is on, else the goal
  formation  the drone's slot from the formation manager
  flock / search  none
Optionally wrapped by ORCA like any other controller (`--rl-orca`).
"""

from __future__ import annotations

import numpy as np

from swarm.agents.state import SwarmState
from swarm.control.base import StepContext
from swarm.rl.observe import Observer, action_to_accel
from swarm.rl.ppo import load_policy


class RLController:
    def __init__(self, policy_path: str, max_speed: float, max_accel: float, perception: float,
                 maps=None, world_size=None):
        self.actor, self.meta = load_policy(policy_path)
        self.task = self.meta.get("task", "navigate")
        self.max_accel = max_accel
        self.max_speed = max_speed
        self.observer = Observer(max_speed, perception, maps, world_size)
        self.maps = maps

    def compute(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        targets = None
        if self.task == "navigate" and ctx.goal is not None:
            targets = ctx.targets if ctx.targets is not None else np.repeat(ctx.goal[None], state.n, 0)
        elif self.task == "formation" and "formation" in ctx.extras:
            ft = ctx.extras["formation"]
            targets = ft.slot.copy()
            # the leader isn't trained to lead: it flies to its aim like the hand-written leader
        obs = self.observer(state.pos, state.vel, targets, ctx.scan,
                            self.maps if self.task == "search" else None)
        action = self.actor.act(obs)
        if self.task == "formation" and "formation" in ctx.extras:
            ft = ctx.extras["formation"]
            L = ft.leader
            to = ft.leader_aim - state.pos[L]
            action[L] = to / max(np.linalg.norm(to), 1e-6) * ft.leader_speed / self.max_speed
        return action_to_accel(action, state.vel, self.max_speed, self.max_accel)
