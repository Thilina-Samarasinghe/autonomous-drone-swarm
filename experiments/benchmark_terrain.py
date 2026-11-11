"""Route planning over terrain and threats: PSO vs hybrid vs grid vs straight.

For each seed and swarm size, the same terrain (hills, no-fly peaks, threat
zones) and the same starts/goals (goals shuffled so routes must cross) are
planned with every planner, then FLOWN in the simulator with ORCA on.

  python -m experiments.benchmark_terrain                 # 6/12/30 drones, 3 seeds
  python -m experiments.benchmark_terrain --sizes 12 --seeds 2

Writes outputs/benchmarks/terrain.csv and terrain.md.
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

PLANNERS = ("straight", "grid", "pso", "hybrid")


def run_one(job: tuple[str, int, int, int]) -> dict:
    planner, seed, drones, steps = job
    sim = Simulation(load_config(None, {"drones": {"count": drones}, "sim": {"seed": seed},
                                        "scenario": {"name": "terrain"},
                                        "terrain": {"planner": planner}}))
    plan = sim.route_plan
    minsep = float("inf")
    t0 = time.perf_counter()
    for _ in range(steps):
        minsep = min(minsep, sim.step().min_separation)
    fl = sim.follower.stats
    sc = plan.scores
    arrived = np.isfinite(fl.arrival)
    return {
        "planner": planner, "seed": seed, "drones": drones,
        "planning_s": plan.seconds,
        "planned_length": sum(s["length"] for s in sc),
        "planned_threat": sum(s["threat"] for s in sc),
        "planned_forbidden_m": sum(s["lethal"] + s["too_high"] for s in sc),
        "planned_conflict": sum(s["conflict"] for s in sc),
        "flown_threat": float(fl.threat.sum()),
        "flown_lethal_s": float(fl.lethal_time.sum()),
        "flown_above_ceiling_s": float(fl.high_time.sum()),
        "close_pair_s": float(fl.close_pair_seconds),
        "mean_delay_s": float(np.mean(plan.delays)),
        "arrived_pct": 100.0 * arrived.mean(),
        "last_arrival_s": float(np.nanmax(fl.arrival)) if arrived.any() else float("nan"),
        "min_separation": minsep,
        "collision_steps": sim.total_collisions,
        "obstacle_hit_steps": sim.total_obstacle_hits,
        "fly_steps_per_s": steps / (time.perf_counter() - t0),
    }


def summarise(rows: list[dict], sizes) -> str:
    lines = [
        "| Drones | Planner | Planning time | Route length (sum) | Forbidden (planned m) | Conflicts (planned) "
        "| Close encounters flown | Threat flown | Lethal core flown | Arrived | Last arrival | Collisions |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in sizes:
        for p in PLANNERS:
            g = [r for r in rows if r["drones"] == n and r["planner"] == p]
            if not g:
                continue
            last = [r["last_arrival_s"] for r in g if not np.isnan(r["last_arrival_s"])]
            lines.append(
                f"| {n} | {p} | {np.mean([r['planning_s'] for r in g]):.1f} s "
                f"| {np.mean([r['planned_length'] for r in g]):.0f} m "
                f"| {np.mean([r['planned_forbidden_m'] for r in g]):.1f} "
                f"| {np.mean([r['planned_conflict'] for r in g]):.0f} "
                f"| {np.mean([r['close_pair_s'] for r in g]):.1f} pair-s "
                f"| {np.mean([r['flown_threat'] for r in g]):.1f} "
                f"| {np.mean([r['flown_lethal_s'] for r in g]):.1f} s "
                f"| {np.mean([r['arrived_pct'] for r in g]):.0f}% "
                f"| {(f'{np.mean(last):.0f} s' if last else 'n/a')} "
                f"| {np.mean([r['collision_steps'] for r in g]):.1f} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="6,12,30")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=2400, help="flight length: 2400 steps = 120 s")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", default="outputs/benchmarks")
    args = p.parse_args(argv)
    sizes = [int(x) for x in args.sizes.split(",")]

    jobs = [(pl, seed, n, args.steps) for n in sizes for pl in PLANNERS for seed in range(1, args.seeds + 1)]
    print(f"Running {len(jobs)} plan+fly runs on {args.workers} workers ...", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, row in enumerate(ex.map(run_one, jobs), 1):
            rows.append(row)
            print(f"  [{k}/{len(jobs)}] {row['drones']:2d} drones {row['planner']:8s} seed {row['seed']}: "
                  f"conflict={row['planned_conflict']:.0f} lethal={row['flown_lethal_s']:.1f}s "
                  f"arrived={row['arrived_pct']:.0f}% coll={row['collision_steps']}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "terrain.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = summarise(rows, sizes)
    header = (f"# Terrain route-planning benchmark\n\n{args.seeds} seeds per row; routes flown for "
              f"{args.steps * 0.05:.0f} s with ORCA on.\n\n")
    (out / "terrain.md").write_text(header + table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
