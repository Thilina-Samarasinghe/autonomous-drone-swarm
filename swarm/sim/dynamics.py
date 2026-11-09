"""Point-mass dynamics with acceleration and speed limits (semi-implicit Euler)."""

from __future__ import annotations

import numpy as np

from swarm.agents.state import SwarmState
from swarm.mathutil import clamp_norm
from swarm.sim.world import World


def integrate(
    state: SwarmState,
    accel: np.ndarray,
    dt: float,
    max_speed: float,
    max_accel: float,
    world: World,
    radius: float,
) -> np.ndarray:
    """Advance the state in place. Returns the acceleration actually applied."""
    a = clamp_norm(accel, max_accel)
    state.vel = clamp_norm(state.vel + a * dt, max_speed)
    state.pos = state.pos + state.vel * dt
    world.enforce_bounds(state.pos, state.vel, radius)
    return a
