"""Make the key result figures for the README (docs/figures/*.png).

Reads the CSVs the benchmark scripts write to outputs/benchmarks/ (run those
first), plus one quick route-planning run for the terrain map.

  python -m experiments.make_figures

Figures (only the most important result of each phase):
  avoidance.png        Phase 2: which avoidance method keeps drones off obstacles
  comms_coverage.png   Phase 3: how fast each drone learns the map, by radio setting
  planning.png         Phase 4: planning vs no planning in the maze
  rescue.png           Phase 6: victims found over time, by strategy
  terrain.png          Phase 7: grid vs hybrid-PSO routes, and close encounters
  rl.png               Phase 8: learned vs hand-written navigation
  robustness.png       Phase 9: scaling, radio loss and sensor noise
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BENCH = Path("outputs/benchmarks")
OUT = Path("docs/figures")
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096", "#6b46c1"
METHOD_COLOURS = {"none": GREY, "apf": ORANGE, "orca": GREEN, "hybrid": BLUE}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 120})


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (skipped: {path} not found - run its benchmark first)")
        return []
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def _mean(rows, key):
    v = [r[key] for r in rows if isinstance(r[key], float) and not np.isnan(r[key])]
    return float(np.mean(v)) if v else float("nan")


def _label_bars(ax, bars, fmt="{:.0f}", zero_text="0"):
    for b in bars:
        h = b.get_height()
        txt = zero_text if h == 0 else fmt.format(h)
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, max(h, ax.get_ylim()[0])),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


# ------------------------------------------------------------------ phase 2
def avoidance():
    rows = _rows(BENCH / "avoidance.csv")
    if not rows:
        return
    scen = ["forest", "corridor", "random"]
    methods = [m for m in ("none", "apf", "orca", "hybrid") if any(r["method"] == m for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        hits = [_mean([r for r in rows if r["scenario"] == s and r["method"] == m], "obstacle_hit_steps") for s in scen]
        t90 = [_mean([r for r in rows if r["scenario"] == s and r["method"] == m], "t90") for s in scen]
        x = np.arange(len(scen)) + (i - (len(methods) - 1) / 2) * w
        b = axes[0].bar(x, np.maximum(hits, 0.5), w, color=METHOD_COLOURS[m], label=m)
        for bar, h in zip(b, hits):
            axes[0].annotate("0" if h == 0 else f"{h:.0f}", (bar.get_x() + bar.get_width() / 2, max(h, 0.5)),
                             ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
        axes[1].bar(x, t90, w, color=METHOD_COLOURS[m], label=m)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("obstacle contact (drone-steps, log)")
    axes[0].set_title("Obstacle contact - lower is better")
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Time for 90% of the swarm to reach the goal")
    for ax in axes:
        ax.set_xticks(range(len(scen)), scen)
    axes[0].legend(title="avoidance", ncol=2, fontsize=8)
    fig.suptitle("Phase 2 - 50 drones crossing obstacle fields (5 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "avoidance.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ phase 4
def planning():
    rows = _rows(BENCH / "planning.csv")
    if not rows:
        return
    labels = list(dict.fromkeys(r["setting"] for r in rows))
    arrived = [_mean([r for r in rows if r["setting"] == s], "arrived_pct") for s in labels]
    t90 = [_mean([r for r in rows if r["setting"] == s], "t90") for s in labels]
    cols = [GREY, ORANGE, GREEN, BLUE][:len(labels)]
    names = [lbl.replace(", ", "\n").replace(" + ", "\n+ ") for lbl in labels]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    b = axes[0].bar(names, arrived, color=cols)
    _label_bars(axes[0], b, "{:.0f}%")
    axes[0].set_ylim(0, 115)
    axes[0].set_title("Drones that escaped the maze")
    axes[0].set_ylabel("%")
    t90p = [0 if np.isnan(t) else t for t in t90]
    b = axes[1].bar(names, t90p, color=cols)
    for bar, t in zip(b, t90):
        axes[1].annotate("never" if np.isnan(t) else f"{t:.0f} s", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    axes[1].set_title("Time for 90% to reach the goal - lower is better")
    axes[1].set_ylabel("seconds")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Phase 4 - 30 drones in the maze, each planning on its own map (3 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "planning.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ phase 7
def terrain():
    rows = _rows(BENCH / "terrain.csv")
    from swarm.config import TerrainConfig
    from swarm.sim.scenarios import build_scenario
    from swarm.terrain.route_planner import plan_routes
    from matplotlib.patches import Circle

    W = np.array([200.0, 120.0])
    cfg = TerrainConfig()
    sc = build_scenario("terrain", W, np.random.default_rng(4), num_drones=10, terrain_cfg=cfg)
    xs, ys = np.meshgrid(np.linspace(0, 200, 201), np.linspace(0, 120, 121))
    H = sc.terrain.height(np.stack([xs, ys], -1))
    fig = plt.figure(figsize=(14, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
    for col, method in enumerate(("grid", "hybrid")):
        ax = fig.add_subplot(gs[0, col])
        plan = plan_routes(method, sc.starts, sc.goals, sc.terrain, cfg, W, 8.0, np.random.default_rng(1))
        ax.imshow(H, origin="lower", extent=(0, 200, 0, 120), cmap="Greens_r", vmin=-60, vmax=110, alpha=0.8)
        ax.contourf(xs, ys, H, levels=[sc.terrain.ceiling, 1e3], colors=["#a0aec0"])
        for cx, cy, r in sc.terrain.threats:
            ax.add_patch(Circle((cx, cy), r, fill=False, color=PURPLE, lw=0.8))
            ax.add_patch(Circle((cx, cy), r * sc.terrain.lethal_fraction, color=RED, alpha=0.5, lw=0))
        cmap = plt.get_cmap("tab10")
        for k, rt in enumerate(plan.routes):
            ax.plot(rt[:, 0], rt[:, 1], lw=1.6, color=cmap(k % 10))
        conflict = sum(s["conflict"] for s in plan.scores)
        ax.set_title(f"{method}: planned conflicts = {conflict:.0f}")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 120)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    if rows:
        sizes = sorted({int(r["drones"]) for r in rows})
        planners = [p for p in ("straight", "grid", "pso", "hybrid") if any(r["planner"] == p for r in rows)]
        cols = {"straight": GREY, "grid": ORANGE, "pso": PURPLE, "hybrid": BLUE}
        w = 0.8 / len(planners)
        for i, p in enumerate(planners):
            v = [_mean([r for r in rows if int(r["drones"]) == n and r["planner"] == p], "close_pair_s") for n in sizes]
            ax.bar(np.arange(len(sizes)) + (i - (len(planners) - 1) / 2) * w, v, w, color=cols[p], label=p)
        ax.set_xticks(range(len(sizes)), [f"{n} drones" for n in sizes])
        ax.set_ylabel("pair-seconds closer than 6 m")
        ax.set_title("Close encounters in flight")
        ax.legend(fontsize=8)
    fig.suptitle("Phase 7 - routes over terrain (grey = too high, red = lethal core, purple = threat zone)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "terrain.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ phase 8
def rl():
    rows = _rows(BENCH / "rl_stage1" / "rl.csv")
    rows = [r for r in rows if r.get("task") == "navigate"]
    if not rows:
        return
    cases = ["boids alone", "boids + ORCA", "RL alone", "RL + ORCA"]
    cols = [GREY, GREEN, ORANGE, BLUE]
    t90 = [_mean([r for r in rows if r["case"].endswith(c)], "t90") for c in cases]
    coll = [_mean([r for r in rows if r["case"].endswith(c)], "collision_steps") for c in cases]
    hits = [_mean([r for r in rows if r["case"].endswith(c)], "obstacle_hit_steps") for c in cases]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    b = axes[0].bar(cases, t90, color=cols)
    _label_bars(axes[0], b, "{:.0f} s")
    axes[0].set_title("Time for 90% to reach the goal - lower is better")
    axes[0].set_ylabel("seconds")
    x = np.arange(len(cases))
    b1 = axes[1].bar(x - 0.2, np.maximum(coll, 0.5), 0.4, color=RED, label="drone collisions")
    b2 = axes[1].bar(x + 0.2, np.maximum(hits, 0.5), 0.4, color=PURPLE, label="obstacle hits")
    axes[1].set_yscale("log")
    for bars, vals in ((b1, coll), (b2, hits)):
        for bar, v in zip(bars, vals):
            axes[1].annotate("0" if v == 0 else f"{v:.0f}", (bar.get_x() + bar.get_width() / 2, max(v, 0.5)),
                             ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    axes[1].set_xticks(x, cases)
    axes[1].set_title("Contact (drone-steps, log) - lower is better")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Phase 8 - learned vs hand-written navigation, 20 drones, forest + random fields (3 seeds each)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "rl.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("avoidance", avoidance), ("planning", planning), ("terrain", terrain), ("rl", rl)):
        print(f"- {name}")
        fn()
    for src, dst in ((BENCH / "comms_coverage.png", "comms_coverage.png"),
                     (BENCH / "rescue_found.png", "rescue.png"),
                     (BENCH / "robustness.png", "robustness.png")):
        if src.exists():
            shutil.copy(src, OUT / dst)
            print(f"- {dst} (copied)")
    print(f"Figures in {OUT.resolve()}")


if __name__ == "__main__":
    main()
