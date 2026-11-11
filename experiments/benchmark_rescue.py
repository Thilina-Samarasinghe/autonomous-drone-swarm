"""Which search strategy finds the victims fastest?

Runs the `rescue` scenario (8 victims hidden among obstacles, drones start at
the base, batteries force trips home) with different strategies:

  random / camera only     fly to random points, find victims by sight
  frontier / camera only   explore the nearest unknown space, by sight
  voronoi / camera only    explore with the area split between drones, by sight
  frontier / PSO           ... plus particle-swarm search on victims' signals
  voronoi / PSO            the full system
  voronoi / PSO, no radio  the full system with the radio switched off

  python -m experiments.benchmark_rescue                  # 20 drones, 3 seeds, 400 s
  python -m experiments.benchmark_rescue --drones 30 --seeds 2 --steps 6000

Writes outputs/benchmarks/rescue.csv, rescue.md and rescue_found.png.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from swarm.config import load_config
from swarm.sim.engine import Simulation

SETTINGS: dict[str, dict] = {
    "random / camera only": {"mission": {"explore": "random", "seek": "none"}},
    "frontier / camera only": {"mission": {"explore": "frontier", "seek": "none"}},
    "voronoi / camera only": {"mission": {"explore": "voronoi", "seek": "none"}},
    "frontier / PSO": {"mission": {"explore": "frontier", "seek": "pso"}},
    "voronoi / PSO": {"mission": {"explore": "voronoi", "seek": "pso"}},
    "voronoi / PSO, no radio": {"mission": {"explore": "voronoi", "seek": "pso"},
                                "comms": {"enabled": False}},
}


def run_one(job: tuple[str, int, int, int]) -> dict:
    label, seed, drones, steps = job
    over = {"drones": {"count": drones}, "sim": {"seed": seed}, "scenario": {"name": "rescue"}}
    over.update(SETTINGS[label])
    sim = Simulation(load_config(None, over))
    t0 = time.perf_counter()
    curve_t, curve_found = [0.0], [0]
    every = max(1, round(5.0 / sim.cfg.sim.dt))
    for _ in range(steps):
        sim.step()
        if sim.step_count % every == 0:
            curve_t.append(sim.time)
            curve_found.append(sim.mission.stats.found)
    wall = time.perf_counter() - t0
    ms = sim.mission.stats
    horizon = steps * sim.cfg.sim.dt
    found_t = ms.found_time
    return {
        "setting": label, "seed": seed, "drones": drones,
        "found": ms.found, "reported": ms.reported, "victims": int(sim.mission.V),
        # unfound victims count as "found at the end of the mission" (a lower bound)
        "mean_find_time_s": float(np.where(np.isnan(found_t), horizon, found_t).mean()),
        "all_found_s": float(np.nanmax(found_t)) if ms.found == sim.mission.V else float("nan"),
        "all_reported_s": ms.complete_time,
        "coverage": sim.info_metrics().map_coverage_team,
        "lost": ms.lost, "returns": ms.returns, "stuck": ms.stuck,
        "collision_steps": sim.total_collisions,
        "obstacle_hit_steps": sim.total_obstacle_hits,
        "steps_per_s": steps / wall,
        "curve_t": curve_t, "curve_found": curve_found,
    }


def _fmt_t(vals):
    ok = [v for v in vals if not np.isnan(v)]
    if not ok:
        return "never"
    s = f"{np.mean(ok):.0f} s"
    return s if len(ok) == len(vals) else s + f" ({len(ok)}/{len(vals)} runs)"


def summarise(rows: list[dict]) -> str:
    lines = [
        "| Strategy | Victims found | Mean time to find a victim | All found | All reported to base "
        "| Map coverage | Drones lost | Collisions | Obstacle hits |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label in SETTINGS:
        g = [r for r in rows if r["setting"] == label]
        if not g:
            continue
        lines.append(
            f"| {label} | {np.mean([r['found'] for r in g]):.1f} / {g[0]['victims']} "
            f"| {np.mean([r['mean_find_time_s'] for r in g]):.0f} s "
            f"| {_fmt_t([r['all_found_s'] for r in g])} "
            f"| {_fmt_t([r['all_reported_s'] for r in g])} "
            f"| {np.mean([r['coverage'] for r in g]):.0%} "
            f"| {np.mean([r['lost'] for r in g]):.1f} "
            f"| {np.mean([r['collision_steps'] for r in g]):.1f} "
            f"| {np.mean([r['obstacle_hit_steps'] for r in g]):.1f} |")
    return "\n".join(lines)


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in SETTINGS:
        g = [r for r in rows if r["setting"] == label]
        if not g:
            continue
        n = min(len(r["curve_t"]) for r in g)
        ax.plot(g[0]["curve_t"][:n], np.mean([r["curve_found"][:n] for r in g], axis=0), lw=1.8, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("victims found (mean over seeds)")
    ax.set_title("Search and rescue: victims found over time")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drones", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=8000, help="8000 steps = 400 s simulated")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", default="outputs/benchmarks")
    args = p.parse_args(argv)

    jobs = [(label, seed, args.drones, args.steps)
            for label in SETTINGS for seed in range(1, args.seeds + 1)]
    print(f"Running {len(jobs)} simulations on {args.workers} workers ...", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, row in enumerate(ex.map(run_one, jobs), 1):
            rows.append(row)
            print(f"  [{k}/{len(jobs)}] {row['setting']:24s} seed {row['seed']}: found {row['found']}/"
                  f"{row['victims']}, lost={row['lost']} coll={row['collision_steps']} "
                  f"hits={row['obstacle_hit_steps']}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in r.items() if not k.startswith("curve")} for r in rows]
    with open(out / "rescue.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    table = summarise(rows)
    header = (f"# Search-and-rescue benchmark\n\n{args.drones} drones, {args.seeds} seeds, "
              f"{args.steps * 0.05:.0f} s missions. Mean time to find counts unfound victims "
              "as found at the end of the mission (a lower bound).\n\n")
    (out / "rescue.md").write_text(header + table + "\n", encoding="utf-8")
    plot(rows, out / "rescue_found.png")
    print("\n" + table)
    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
