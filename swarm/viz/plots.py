"""Offline matplotlib plots: trajectories and metric time-series."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from swarm.metrics import Metrics  # noqa: E402
from swarm.sim.world import World  # noqa: E402


def draw_obstacles(ax, world: World, goal=None, goal_radius: float = 15.0) -> None:
    from matplotlib.patches import Circle, Rectangle
    for cx, cy, r in world.obstacles.circles:
        ax.add_patch(Circle((cx, cy), r, color="0.35", zorder=3))
    for x0, y0, x1, y1 in world.obstacles.boxes:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.35", zorder=3))
    if goal is not None:
        ax.add_patch(Circle(tuple(goal), goal_radius, fill=False, ls="--", color="red", lw=1.2, zorder=4))
        ax.plot(*goal, "r*", ms=12, zorder=5)


def plot_trajectories(positions: np.ndarray, world: World, path: str,
                      goal=None, goal_radius: float = 15.0, title: str | None = None) -> None:
    """positions: (T, N, 2)."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    draw_obstacles(ax, world, goal, goal_radius)
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
    ax.set_title(title or f"Swarm trajectories ({n} drones)  ■ start  ● end")
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


def plot_info(info: list, path: str) -> None:
    """Knowledge over time (Phase 3): map coverage, accuracy, radio graph."""
    t = np.array([i.time for i in info])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.plot(t, [i.map_coverage_team for i in info], lw=1.6, label="team (union)")
    ax.plot(t, [i.map_coverage_mean for i in info], lw=1.6, label="average drone")
    ax.plot(t, [i.map_coverage_min for i in info], lw=1.2, ls="--", label="least-informed drone")
    ax.set_ylim(0, 1.02)
    ax.set_title("Map coverage (fraction of arena)")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(t, [i.map_accuracy for i in info], lw=1.4)
    ax.set_title("Map accuracy (known cells correct)")
    ax = axes[2]
    ax.plot(t, [i.comm_groups for i in info], lw=1.4, label="radio groups")
    ax.plot(t, [i.info_age for i in info], lw=1.4, label="info age (s)")
    ax.set_title("Radio network")
    ax.legend(loc="upper right")
    for ax in axes:
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
