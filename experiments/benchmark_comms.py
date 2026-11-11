"""How much does communication help the swarm learn its environment?

Runs the `survey` scenario (obstacles everywhere, drones start scattered,
no goal) under different radio settings and measures how fast every drone's
own map fills in.

  python -m experiments.benchmark_comms                      # 50 drones, 3 seeds
  python -m experiments.benchmark_comms --drones 30 --seeds 2 --steps 1200

Writes outputs/benchmarks/comms.csv, comms.md and comms_coverage.png.
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

# label -> comms overrides
SETTINGS: dict[str, dict] = {
    "no sharing": {"share_maps": False},
    "single-hop, 0% loss": {"share_maps": True, "relay": False, "packet_loss": 0.0},
    "multi-hop, 0% loss": {"share_maps": True, "relay": True, "packet_loss": 0.0},
    "multi-hop, 20% loss": {"share_maps": True, "relay": True, "packet_loss": 0.2},
    "multi-hop, 50% loss": {"share_maps": True, "relay": True, "packet_loss": 0.5},
    "multi-hop, 80% loss": {"share_maps": True, "relay": True, "packet_loss": 0.8},
}


def run_one(job: tuple[str, int, int, int]) -> dict:
    label, seed, drones, steps = job
    cfg = load_config(None, {
        "drones": {"count": drones},
        "sim": {"seed": seed},
        "scenario": {"name": "survey"},
        "comms": SETTINGS[label],
    })
    sim = Simulation(cfg)
    every = max(1, round(1.0 / cfg.sim.dt))
    curve = [sim.info_metrics()]
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
        if sim.step_count % every == 0:
            curve.append(sim.info_metrics())
    wall = time.perf_counter() - t0

    def first(th):
        return next((i.time for i in curve if i.map_coverage_mean >= th), float("nan"))

    last = curve[-1]
    return {
        "setting": label, "seed": seed, "drones": drones,
        "t_mean_50": first(0.5), "t_mean_80": first(0.8), "t_mean_90": first(0.9),
        "cov_mean_30s": next(i.map_coverage_mean for i in curve if i.time >= 30 - 1e-6),
        "cov_mean_final": last.map_coverage_mean,
        "cov_min_final": last.map_coverage_min,
        "cov_team_final": last.map_coverage_team,
        "accuracy": last.map_accuracy,
        "info_age_avg": float(np.nanmean([i.info_age for i in curve[1:]])),
        "kbytes_per_s": last.kbytes_sent / max(sim.time, 1e-9),
        "collisions": sim.total_collisions,
        "obstacle_hits": sim.total_obstacle_hits,
        "steps_per_s": steps / wall,
        "curve_t": [i.time for i in curve],
        "curve_mean": [i.map_coverage_mean for i in curve],
    }


def summarise(rows: list[dict]) -> str:
    def fmt_t(vals):
        ok = [v for v in vals if not np.isnan(v)]
        if not ok:
            return "never"
        s = f"{np.mean(ok):.0f}"
        return s if len(ok) == len(vals) else s + f" ({len(ok)}/{len(vals)})"

    lines = [
        "| Setting | Avg drone map @30 s | Time to 50% (s) | Time to 80% (s) | Time to 90% (s) "
        "| Final avg / worst drone | Accuracy | Info age (s) | Traffic (kB/s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label in SETTINGS:
        g = [r for r in rows if r["setting"] == label]
        if not g:
            continue
        lines.append(
            f"| {label} | {np.mean([r['cov_mean_30s'] for r in g]):.0%} "
            f"| {fmt_t([r['t_mean_50'] for r in g])} | {fmt_t([r['t_mean_80'] for r in g])} "
            f"| {fmt_t([r['t_mean_90'] for r in g])} "
            f"| {np.mean([r['cov_mean_final'] for r in g]):.0%} / {np.mean([r['cov_min_final'] for r in g]):.0%} "
            f"| {np.mean([r['accuracy'] for r in g]):.1%} "
            f"| {np.mean([r['info_age_avg'] for r in g]):.1f} "
            f"| {np.mean([r['kbytes_per_s'] for r in g]):.0f} |")
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
        t = g[0]["curve_t"][:n]
        y = np.mean([r["curve_mean"][:n] for r in g], axis=0)
        ax.plot(t, y, lw=1.8, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("average drone's map coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title("How fast each drone learns the arena (survey scenario)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drones", type=int, default=50)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=2400, help="2400 steps = 120 s simulated")
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
            print(f"  [{k}/{len(jobs)}] {row['setting']:22s} seed {row['seed']}: "
                  f"avg map {row['cov_mean_final']:.0%}, coll={row['collisions']} "
                  f"hits={row['obstacle_hits']}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in r.items() if not k.startswith("curve")} for r in rows]
    with open(out / "comms.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    table = summarise(rows)
    header = (f"# Communication benchmark\n\nScenario `survey`, {args.drones} drones, "
              f"{args.seeds} seeds, {args.steps} steps ({args.steps * 0.05:.0f} s simulated).\n"
              "Coverage = fraction of the arena's cells a drone has classified "
              "(free / occupied) in its *own* map.\n\n")
    (out / "comms.md").write_text(header + table + "\n", encoding="utf-8")
    plot(rows, out / "comms_coverage.png")
    print("\n" + table)
    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
