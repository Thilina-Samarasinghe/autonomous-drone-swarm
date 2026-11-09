"""Offline matplotlib plots: trajectories and metric time-series."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from swarm.metrics import Metrics  # noqa: E402
from swarm.sim.world import World  # noqa: E402


def plot_trajectories(positions: np.ndarray, world: World, path: str) -> None:
    """positions: (T, N, 2)."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    n = positions.shape[1]
    cmap = plt.get_cmap("turbo")
    for k in range(n):
        c = cmap(k / max(n - 1, 1))
        ax.plot(positions[:, k, 0], positions[:, k, 1], lw=0.9, color=c, alpha=0.85)
        ax.plot(*positions[0, k], "s", ms=3, color=c)
        ax.plot(*positions[-1, k], "o", ms=4, color=c)
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Swarm trajectories ({n} drones)  ■ start  ● end")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metrics(history: list[Metrics], radius: float, path: str) -> None:
    t = np.array([m.time for m in history])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    series = [
        ("min_separation", "Min separation [m]", axes[0, 0]),
        ("polarization", "Polarization (alignment order)", axes[0, 1]),
        ("mean_speed", "Mean speed [m/s]", axes[1, 0]),
        ("groups", "Connected groups", axes[1, 1]),
    ]
    for key, label, ax in series:
        ax.plot(t, [getattr(m, key) for m in history], lw=1.2)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0, 0].axhline(2 * radius, color="r", ls="--", lw=1, label="collision distance")
    axes[0, 0].legend(loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
