"""Learned policies vs the hand-written controllers, in the normal simulator.

Policies were trained with 8 drones; here they fly 20 (does parameter
sharing generalise to a bigger swarm?).

  navigate   forest and random obstacle fields, shared goal
             boids alone | boids + ORCA (default) | RL alone | RL + ORCA
  flock      open arena: boids vs RL
  formation  V formation in the open: hand-written slot tracking vs RL
  search     survey scenario, radio + map sharing on: boids flocking vs RL

  python -m experiments.benchmark_rl                   # 20 drones, 3 seeds
  python -m experiments.benchmark_rl --tasks navigate --seeds 2

Writes outputs/benchmarks/rl.csv and rl.md.
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


def _case(task, label, scenario, steps, over):
    return {"task": task, "label": label, "scenario": scenario, "steps": steps, "over": over}


def cases(models: str) -> list[dict]:
    nav, flock = f"{models}/navigate.pt", f"{models}/flock.pt"
    form, search = f"{models}/formation.pt", f"{models}/search.pt"
    out = []
    for sc in ("forest", "random"):
        out += [
            _case("navigate", f"{sc} / boids alone", sc, 2400, {"avoidance": {"method": "none"}}),
            _case("navigate", f"{sc} / boids + ORCA", sc, 2400, {}),
            _case("navigate", f"{sc} / RL alone", sc, 2400, {"rl": {"policy": nav}}),
            _case("navigate", f"{sc} / RL + ORCA", sc, 2400, {"rl": {"policy": nav, "orca": True}}),
        ]
    out += [
        _case("flock", "open / boids", "open", 2400, {"avoidance": {"method": "none"}}),
        _case("flock", "open / RL", "open", 2400, {"rl": {"policy": flock}}),
        _case("formation", "open V / hand-written", "open", 2400,
              {"formation": {"enabled": True, "shape": "v"}}),
        _case("formation", "open V / RL", "open", 2400,
              {"formation": {"enabled": True, "shape": "v"}, "rl": {"policy": form}}),
        _case("search", "survey / boids flocking", "survey", 1200, {}),
        _case("search", "survey / RL", "survey", 1200, {"rl": {"policy": search}}),
    ]
    return out


def run_one(job: tuple[dict, int, int]) -> dict:
    case, seed, drones = job
    import torch
    torch.set_num_threads(1)
    over = {"drones": {"count": drones}, "sim": {"seed": seed}, "scenario": {"name": case["scenario"]}}
    for k, v in case["over"].items():
        over.setdefault(k, {}).update(v)
    sim = Simulation(load_config(None, over))
    hist = sim.run(case["steps"])
    tail = hist[len(hist) // 2:]
    row = {"task": case["task"], "case": case["label"], "seed": seed, "drones": drones,
           "collision_steps": sim.total_collisions, "obstacle_hit_steps": sim.total_obstacle_hits,
           "min_separation": min(m.min_separation for m in hist)}
    if case["task"] == "navigate":
        row["arrived_pct"] = 100.0 * hist[-1].arrived / drones
        row["t90"] = sim.time_to_arrive_90 if sim.time_to_arrive_90 is not None else float("nan")
    elif case["task"] == "flock":
        row["local_alignment"] = float(np.mean([m.local_polarization for m in tail]))
        row["groups"] = float(np.mean([m.groups for m in tail]))
        row["speed"] = float(np.mean([m.mean_speed for m in tail]))
    elif case["task"] == "formation":
        row["slot_error"] = sim.formation.stats.mean_error
        row["in_shape_pct"] = 100.0 * sim.formation.stats.in_shape_fraction
    elif case["task"] == "search":
        im = sim.info_metrics()
        row["coverage_mean"] = im.map_coverage_mean
        row["coverage_team"] = im.map_coverage_team
    return row


def summarise(rows: list[dict]) -> str:
    out = []
    groups = {"navigate": ("arrived_pct", "t90"), "flock": ("local_alignment", "groups", "speed"),
              "formation": ("slot_error", "in_shape_pct"), "search": ("coverage_mean", "coverage_team")}
    for task, keys in groups.items():
        g = [r for r in rows if r["task"] == task]
        if not g:
            continue
        out.append(f"\n**{task}**\n")
        out.append("| Case | " + " | ".join(keys) + " | Drone collisions | Obstacle hits | Closest drones (m) |")
        out.append("|---|" + "---|" * (len(keys) + 3))
        for label in dict.fromkeys(r["case"] for r in g):
            c = [r for r in g if r["case"] == label]
            vals = []
            for k in keys:
                v = [r[k] for r in c if not np.isnan(r[k])]
                vals.append("never" if not v else f"{np.mean(v):.2f}" + ("" if len(v) == len(c) else f" ({len(v)}/{len(c)})"))
            out.append(f"| {label} | " + " | ".join(vals)
                       + f" | {np.mean([r['collision_steps'] for r in c]):.1f}"
                       + f" | {np.mean([r['obstacle_hit_steps'] for r in c]):.1f}"
                       + f" | {min(r['min_separation'] for r in c):.2f} |")
    return "\n".join(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drones", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--tasks", default="navigate,flock,formation,search")
    p.add_argument("--models", default="models")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--out", default="outputs/benchmarks")
    args = p.parse_args(argv)
    tasks = set(args.tasks.split(","))
    jobs = [(c, s, args.drones) for c in cases(args.models) if c["task"] in tasks
            for s in range(1, args.seeds + 1)]
    print(f"Running {len(jobs)} simulations on {args.workers} workers ...", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, row in enumerate(ex.map(run_one, jobs), 1):
            rows.append(row)
            print(f"  [{k}/{len(jobs)}] {row['case']:28s} seed {row['seed']}: coll={row['collision_steps']} "
                  f"hits={row['obstacle_hit_steps']}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out / "rl.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    table = summarise(rows)
    (out / "rl.md").write_text(f"# Learned vs hand-written\n\n{args.drones} drones, {args.seeds} seeds.\n"
                               + table + "\n", encoding="utf-8")
    print(table)
    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
