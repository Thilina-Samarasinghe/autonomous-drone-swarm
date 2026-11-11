"""Compare avoidance methods across scenarios and seeds.

  python -m experiments.benchmark_avoidance                  # full: 50 drones, 5 seeds
  python -m experiments.benchmark_avoidance --drones 30 --seeds 3 --steps 1500
  python -m experiments.benchmark_avoidance --scenarios forest --methods orca,hybrid

Writes outputs/benchmarks/avoidance.csv and avoidance.md (a results table).
Runs in parallel worker processes.
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
from swarm.control.factory import METHODS
from swarm.sim.engine import Simulation

DEFAULT_SCENARIOS = ("forest", "corridor", "random")


def run_one(job: tuple[str, str, int, int, int]) -> dict:
    scenario, method, seed, drones, steps = job
    cfg = load_config(None, {
        "drones": {"count": drones},
        "sim": {"seed": seed},
        "scenario": {"name": scenario},
        "avoidance": {"method": method},
    })
    sim = Simulation(cfg)
    t0 = time.perf_counter()
    history = sim.run(steps)
    wall = time.perf_counter() - t0
    return {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "drones": drones,
        "collision_steps": sim.total_collisions,
        "obstacle_hit_steps": sim.total_obstacle_hits,
        "min_separation": min(m.min_separation for m in history),
        "min_clearance": min(m.min_clearance for m in history),
        "arrived_pct": 100.0 * history[-1].arrived / drones,
        "t90": sim.time_to_arrive_90 if sim.time_to_arrive_90 is not None else float("nan"),
        "steps_per_s": steps / wall,
    }


def summarise(rows: list[dict]) -> str:
    lines = [
        "| Scenario | Method | Runs | Collision-free runs | Drone collisions (avg) "
        "| Obstacle hits (avg) | Closest drones (m) | Min clearance (m) | Arrived % "
        "| Time to 90% (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    keys = sorted({(r["scenario"], r["method"]) for r in rows},
                  key=lambda k: (k[0], METHODS.index(k[1])))
    for sc, m in keys:
        g = [r for r in rows if r["scenario"] == sc and r["method"] == m]
        clean = sum(1 for r in g if r["collision_steps"] == 0 and r["obstacle_hit_steps"] == 0)
        t90 = [r["t90"] for r in g if not np.isnan(r["t90"])]
        t90s = f"{np.mean(t90):.1f}" + ("" if len(t90) == len(g) else f" ({len(t90)}/{len(g)} runs)") \
            if t90 else "never"
        lines.append(
            f"| {sc} | {m} | {len(g)} | {clean}/{len(g)} "
            f"| {np.mean([r['collision_steps'] for r in g]):.1f} "
            f"| {np.mean([r['obstacle_hit_steps'] for r in g]):.1f} "
            f"| {min(r['min_separation'] for r in g):.2f} "
            f"| {min(r['min_clearance'] for r in g):.2f} "
            f"| {np.mean([r['arrived_pct'] for r in g]):.0f} "
            f"| {t90s} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drones", type=int, default=50)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=2400, help="2400 steps = 120 s simulated")
    p.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", default="outputs/benchmarks")
    args = p.parse_args(argv)

    jobs = [(sc, m, seed, args.drones, args.steps)
            for sc in args.scenarios.split(",")
            for m in args.methods.split(",")
            for seed in range(1, args.seeds + 1)]
    print(f"Running {len(jobs)} simulations on {args.workers} workers ...", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = []
        for k, row in enumerate(ex.map(run_one, jobs), 1):
            rows.append(row)
            print(f"  [{k}/{len(jobs)}] {row['scenario']:9s} {row['method']:7s} seed {row['seed']}: "
                  f"coll={row['collision_steps']} hits={row['obstacle_hit_steps']} "
                  f"arrived={row['arrived_pct']:.0f}%", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "avoidance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = summarise(rows)
    header = (f"# Avoidance benchmark\n\n{args.drones} drones, {args.seeds} seeds, "
              f"{args.steps} steps ({args.steps * 0.05:.0f} s simulated) per run.\n"
              "Counts are drone-steps in contact (a single touch lasting 5 steps counts 5).\n\n")
    (out / "avoidance.md").write_text(header + table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
