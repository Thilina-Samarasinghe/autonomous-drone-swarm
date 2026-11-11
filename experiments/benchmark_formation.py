"""How well do formations hold, and can they get through obstacles?

Cases
  open      no goal: the leader flies a patrol loop; every shape is measured
  corridor  a 14 m gap: V and line with squeezing on, V with squeezing off
  forest    pillar grid, V
  maze      S-shaped maze with path planning, V
  baseline  free flocking (no formation) in the corridor, for comparison

  python -m experiments.benchmark_formation                 # 20 drones, 3 seeds
  python -m experiments.benchmark_formation --drones 30 --seeds 2

Writes outputs/benchmarks/formation.csv and formation.md.
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

STRANDED = 20.0      # a follower this far from its slot (or trail point) at the end is left behind

# label -> (scenario, steps, overrides)
CASES: dict[str, tuple[str, int, dict]] = {
    "open / line": ("open", 2400, {"formation": {"enabled": True, "shape": "line"}}),
    "open / v": ("open", 2400, {"formation": {"enabled": True, "shape": "v"}}),
    "open / circle": ("open", 2400, {"formation": {"enabled": True, "shape": "circle"}}),
    "open / box": ("open", 2400, {"formation": {"enabled": True, "shape": "box"}}),
    "corridor / flocking (no formation)": ("corridor", 3000, {}),
    "corridor / v, squeeze on": ("corridor", 3000, {"formation": {"enabled": True, "shape": "v"}}),
    "corridor / v, squeeze off": ("corridor", 3000,
                                  {"formation": {"enabled": True, "shape": "v", "adaptive": False}}),
    "corridor / line, squeeze on": ("corridor", 3000, {"formation": {"enabled": True, "shape": "line"}}),
    "forest / v": ("forest", 3000, {"formation": {"enabled": True, "shape": "v"}}),
    "maze / v + planning": ("maze", 5000, {"formation": {"enabled": True, "shape": "v"},
                                           "planning": {"enabled": True}}),
}


def run_one(job: tuple[str, int, int]) -> dict:
    label, seed, drones = job
    scenario, steps, extra = CASES[label]
    over = {"drones": {"count": drones}, "sim": {"seed": seed}, "scenario": {"name": scenario}}
    over.update(extra)
    sim = Simulation(load_config(None, over))
    t0 = time.perf_counter()
    sim.run(steps)
    wall = time.perf_counter() - t0
    fs = sim.formation.stats if sim.formation else None
    leader_arrived = sim.time_leader_arrived if sim.time_leader_arrived is not None else float("nan")
    if sim.formation is not None:
        far = sim.formation.follower_errors(sim.state.pos) > STRANDED
    else:   # free flocking has no slots: count drones far from the swarm centre
        far = np.linalg.norm(sim.state.pos - sim.state.pos.mean(0), axis=1) > 2 * STRANDED
    return {
        "case": label, "seed": seed, "drones": drones,
        "slot_error_m": fs.mean_error if fs else float("nan"),
        "in_shape_pct": 100 * fs.in_shape_fraction if fs else float("nan"),
        "squeezes": fs.squeezes if fs else 0,
        "leader_arrived_s": leader_arrived,
        "stranded": int(far.sum()),
        "collision_steps": sim.total_collisions,
        "obstacle_hit_steps": sim.total_obstacle_hits,
        "steps_per_s": steps / wall,
    }


def _fmt(vals, fmt, unit=""):
    ok = [v for v in vals if not np.isnan(v)]
    if not ok:
        return "n/a" if all(np.isnan(v) for v in vals) else "never"
    s = format(np.mean(ok), fmt) + unit
    return s if len(ok) == len(vals) else s + f" ({len(ok)}/{len(vals)} runs)"


def summarise(rows: list[dict]) -> str:
    lines = [
        "| Case | Mean slot error | Time in shape | Squeezes | Leader reached goal "
        "| Followers left behind | Drone collisions | Obstacle hits |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in CASES:
        g = [r for r in rows if r["case"] == label]
        if not g:
            continue
        lines.append(
            f"| {label} | {_fmt([r['slot_error_m'] for r in g], '.1f', ' m')} "
            f"| {_fmt([r['in_shape_pct'] for r in g], '.0f', '%')} "
            f"| {np.mean([r['squeezes'] for r in g]):.1f} "
            f"| {_fmt([r['leader_arrived_s'] for r in g], '.0f', ' s') if CASES[label][0] != 'open' else 'n/a (patrol)'} "
            f"| {np.mean([r['stranded'] for r in g]):.1f} "
            f"| {np.mean([r['collision_steps'] for r in g]):.1f} "
            f"| {np.mean([r['obstacle_hit_steps'] for r in g]):.1f} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drones", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", default="outputs/benchmarks")
    args = p.parse_args(argv)

    jobs = [(label, seed, args.drones) for label in CASES for seed in range(1, args.seeds + 1)]
    print(f"Running {len(jobs)} simulations on {args.workers} workers ...", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, row in enumerate(ex.map(run_one, jobs), 1):
            rows.append(row)
            print(f"  [{k}/{len(jobs)}] {row['case']:36s} seed {row['seed']}: "
                  f"err={row['slot_error_m']:.1f} stranded={row['stranded']} "
                  f"coll={row['collision_steps']} hits={row['obstacle_hit_steps']}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "formation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = summarise(rows)
    header = (f"# Formation benchmark\n\n{args.drones} drones, {args.seeds} seeds per case. "
              f"\"Left behind\" = followers more than {STRANDED:.0f} m from their slot (or from their "
              "point on the leader's trail, in a column) at the end.\n\n")
    (out / "formation.md").write_text(header + table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
