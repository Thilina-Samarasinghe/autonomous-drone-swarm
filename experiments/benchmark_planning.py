"""Does path planning get the swarm through a maze, and does sharing maps help?

Runs the `maze` scenario under four settings:
  no planning           boids + ORCA flying straight at the goal (baseline)
  plan, own map only    every drone plans on what it has seen itself
  plan + shared maps    drones also merge what teammates discovered (Phase 3)
  A* + shared maps      same, with the A* planner instead of the cost field

  python -m experiments.benchmark_planning                    # 30 drones, 3 seeds
  python -m experiments.benchmark_planning --drones 50 --seeds 2

Writes outputs/benchmarks/planning.csv and planning.md.
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
    "no planning": {"planning": {"enabled": False}},
    "plan, own map only": {"planning": {"enabled": True, "algorithm": "field"},
                           "comms": {"share_maps": False}},
    "plan + shared maps": {"planning": {"enabled": True, "algorithm": "field"},
                           "comms": {"share_maps": True}},
    "A* + shared maps": {"planning": {"enabled": True, "algorithm": "astar"},
                         "comms": {"share_maps": True}},
}


def run_one(job: tuple[str, int, int, int]) -> dict:
    label, seed, drones, steps = job
    over = {"drones": {"count": drones}, "sim": {"seed": seed}, "scenario": {"name": "maze"}}
    over.update(SETTINGS[label])
    sim = Simulation(load_config(None, over))
    t0 = time.perf_counter()
    history = sim.run(steps)
    wall = time.perf_counter() - t0
    ps = sim.planner.stats if sim.planner else None
    return {
        "setting": label, "seed": seed, "drones": drones,
        "arrived_pct": 100.0 * history[-1].arrived / drones,
        "t90": sim.time_to_arrive_90 if sim.time_to_arrive_90 is not None else float("nan"),
        "collision_steps": sim.total_collisions,
        "obstacle_hit_steps": sim.total_obstacle_hits,
        "plans": ps.plans if ps else 0,
        "plan_ms": ps.mean_ms if ps else float("nan"),
        "blocked_replans": ps.blocked_replans if ps else 0,
        "failed_plans": ps.failed if ps else 0,
        "steps_per_s": steps / wall,
    }


def summarise(rows: list[dict]) -> str:
    lines = [
        "| Setting | Arrived | Time for 90% to arrive (s) | Drone collisions | Obstacle hits "
        "| Plans per drone | ms per plan | Sim speed (steps/s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in SETTINGS:
        g = [r for r in rows if r["setting"] == label]
        if not g:
            continue
        t = [r["t90"] for r in g if not np.isnan(r["t90"])]
        t_s = ("never" if not t else f"{np.mean(t):.0f}"
               + ("" if len(t) == len(g) else f" ({len(t)}/{len(g)} runs)"))
        ms = [r["plan_ms"] for r in g if not np.isnan(r["plan_ms"])]
        lines.append(
            f"| {label} | {np.mean([r['arrived_pct'] for r in g]):.0f}% | {t_s} "
            f"| {np.mean([r['collision_steps'] for r in g]):.1f} "
            f"| {np.mean([r['obstacle_hit_steps'] for r in g]):.1f} "
            f"| {np.mean([r['plans'] / r['drones'] for r in g]):.0f} "
            f"| {np.mean(ms):.1f} | {np.mean([r['steps_per_s'] for r in g]):.0f} |"
            if ms else
            f"| {label} | {np.mean([r['arrived_pct'] for r in g]):.0f}% | {t_s} "
            f"| {np.mean([r['collision_steps'] for r in g]):.1f} "
            f"| {np.mean([r['obstacle_hit_steps'] for r in g]):.1f} | n/a | n/a "
            f"| {np.mean([r['steps_per_s'] for r in g]):.0f} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drones", type=int, default=30)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=4000, help="4000 steps = 200 s simulated")
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
            print(f"  [{k}/{len(jobs)}] {row['setting']:20s} seed {row['seed']}: "
                  f"arrived {row['arrived_pct']:.0f}%, t90={row['t90']:.0f}, "
                  f"coll={row['collision_steps']} hits={row['obstacle_hit_steps']}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "planning.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = summarise(rows)
    header = (f"# Planning benchmark\n\nScenario `maze`, {args.drones} drones, {args.seeds} seeds, "
              f"{args.steps} steps ({args.steps * 0.05:.0f} s simulated) per run.\n\n")
    (out / "planning.md").write_text(header + table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
