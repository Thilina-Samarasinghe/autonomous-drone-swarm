"""Phase 9: how gracefully does the swarm degrade?

Four suites, each changing ONE thing at a time:

  scaling   10 / 20 / 30 / 50 drones: forest navigation, V formation through
            the corridor, search and rescue
  radio     packet loss 0-90% and radio range 15-50 m: rescue and V formation
  failures  drones fail mid-mission: 25% / 50% of the rescue swarm at 60 s,
            the formation LEADER (plus 25%) at 20 s, 30% of a navigating swarm
  sensors   lidar noise (0-2 m) and dropped beams (0-60%): forest navigation

  python -m experiments.robustness                      # everything, 2 seeds (~45 min)
  python -m experiments.robustness --suites radio,failures --seeds 3

Writes outputs/benchmarks/robustness.csv, robustness.md and robustness_*.png.
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

RESCUE_STEPS = 6000          # 300 s
NAV_STEPS = 2400             # 120 s
FORM_STEPS = 2400


def _c(suite, kind, label, drones, steps, scenario, over, x=None):
    return dict(suite=suite, kind=kind, label=label, drones=drones, steps=steps,
                scenario=scenario, over=over, x=x)


def cases() -> list[dict]:
    out = []
    for n in (10, 20, 30, 50):
        out.append(_c("scaling", "nav", f"forest navigation, {n} drones", n, NAV_STEPS, "forest", {}, n))
        out.append(_c("scaling", "formation", f"V formation through corridor, {n} drones", n, FORM_STEPS,
                      "corridor", {"formation": {"enabled": True, "shape": "v"}}, n))
        out.append(_c("scaling", "rescue", f"search and rescue, {n} drones", n, RESCUE_STEPS, "rescue", {}, n))
    for loss in (0.0, 0.3, 0.6, 0.9):
        out.append(_c("radio", "rescue", f"rescue, {loss:.0%} packet loss", 20, RESCUE_STEPS, "rescue",
                      {"comms": {"packet_loss": loss}}, loss))
        out.append(_c("radio", "formation", f"V formation, {loss:.0%} packet loss", 20, FORM_STEPS, "open",
                      {"formation": {"enabled": True, "shape": "v"}, "comms": {"packet_loss": loss}}, loss))
    for rng in (15.0, 50.0):
        out.append(_c("radio", "rescue", f"rescue, {rng:.0f} m radio range", 20, RESCUE_STEPS, "rescue",
                      {"comms": {"range": rng}}, rng))
    for frac in (0.0, 0.25, 0.5):
        out.append(_c("failures", "rescue", f"rescue, {frac:.0%} of drones fail at 60 s", 20, RESCUE_STEPS,
                      "rescue", {"faults": {"fail_time": 60.0 if frac else -1.0, "fail_fraction": frac}}, frac))
    out.append(_c("failures", "formation", "V formation, no failure", 20, FORM_STEPS, "open",
                  {"formation": {"enabled": True, "shape": "v"}}, 0))
    out.append(_c("failures", "formation", "V formation, leader + 25% fail at 20 s", 20, FORM_STEPS, "open",
                  {"formation": {"enabled": True, "shape": "v"},
                   "faults": {"fail_time": 20.0, "fail_fraction": 0.25, "fail_leader": True}}, 1))
    out.append(_c("failures", "nav", "forest navigation, 30% fail at 10 s", 20, NAV_STEPS, "forest",
                  {"faults": {"fail_time": 10.0, "fail_fraction": 0.3}}, 0.3))
    for noise in (0.0, 0.5, 1.0, 2.0):
        out.append(_c("sensors", "nav", f"forest navigation, lidar noise {noise} m", 20, NAV_STEPS, "forest",
                      {"faults": {"lidar_noise": noise}}, noise))
    for drop in (0.3, 0.6):
        out.append(_c("sensors", "nav", f"forest navigation, {drop:.0%} beams dropped", 20, NAV_STEPS, "forest",
                      {"faults": {"lidar_dropout": drop}}, drop))
    return out


def run_one(job: tuple[dict, int]) -> dict:
    case, seed = job
    over = {"drones": {"count": case["drones"]}, "sim": {"seed": seed}, "scenario": {"name": case["scenario"]}}
    for k, v in case["over"].items():
        over.setdefault(k, {}).update(v)
    sim = Simulation(load_config(None, over))
    t0 = time.perf_counter()
    hist = sim.run(case["steps"])
    wall = time.perf_counter() - t0
    row = {k: case[k] for k in ("suite", "kind", "label", "drones", "x")}
    row.update(seed=seed, collision_steps=sim.total_collisions, obstacle_hit_steps=sim.total_obstacle_hits,
               min_separation=min(m.min_separation for m in hist), alive=int(sim.alive.sum()),
               steps_per_s=case["steps"] / wall)
    if case["kind"] == "nav":
        row["arrived_pct"] = 100.0 * hist[-1].arrived / max(int(sim.alive.sum()), 1)
        row["t90"] = sim.time_to_arrive_90 if sim.time_to_arrive_90 is not None else float("nan")
    elif case["kind"] == "formation":
        f = sim.formation
        row["slot_error"] = f.stats.mean_error
        row["in_shape_pct"] = 100.0 * f.stats.in_shape_fraction
        row["final_error"] = f.stats.last_error
        row["reelections"] = f.reelections
        row["left_behind"] = int((f.follower_errors(sim.state.pos) > 20.0).sum())
        row["leader_arrived"] = sim.time_leader_arrived if sim.time_leader_arrived is not None else float("nan")
    elif case["kind"] == "rescue":
        ms = sim.mission.stats
        row["found"] = ms.found
        row["reported"] = ms.reported
        row["all_reported_s"] = ms.complete_time
        row["coverage"] = sim.info_metrics().map_coverage_team
        row["battery_lost"] = ms.lost
    return row


FIELDS = {
    "nav": [("arrived_pct", "Arrived (of flying)", "{:.0f}%"), ("t90", "90% arrived by", "{:.0f} s"),
            ("collision_steps", "Collisions", "{:.1f}"), ("obstacle_hit_steps", "Obstacle hits", "{:.1f}"),
            ("min_separation", "Closest drones", "{:.2f} m"), ("steps_per_s", "Sim speed", "{:.0f} steps/s")],
    "formation": [("slot_error", "Mean slot error", "{:.1f} m"), ("final_error", "Error at end", "{:.1f} m"),
                  ("in_shape_pct", "Time in shape", "{:.0f}%"), ("reelections", "Leader re-elections", "{:.0f}"),
                  ("left_behind", "Left behind", "{:.1f}"), ("collision_steps", "Collisions", "{:.1f}"),
                  ("steps_per_s", "Sim speed", "{:.0f} steps/s")],
    "rescue": [("found", "Victims found /8", "{:.1f}"), ("reported", "Reported", "{:.1f}"),
               ("all_reported_s", "All reported by", "{:.0f} s"), ("coverage", "Map coverage", "{:.0%}"),
               ("alive", "Drones flying at end", "{:.0f}"), ("battery_lost", "Battery deaths", "{:.1f}"),
               ("collision_steps", "Collisions", "{:.1f}"), ("steps_per_s", "Sim speed", "{:.0f} steps/s")],
}


def _cell(vals, fmt):
    ok = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    if not ok:
        return "never"
    s = fmt.format(np.mean(ok))
    return s if len(ok) == len(vals) else s + f" ({len(ok)}/{len(vals)})"


def summarise(rows: list[dict]) -> str:
    out = []
    for suite in ("scaling", "radio", "failures", "sensors"):
        srows = [r for r in rows if r["suite"] == suite]
        if not srows:
            continue
        out.append(f"\n## {suite}\n")
        for kind in ("nav", "formation", "rescue"):
            krows = [r for r in srows if r["kind"] == kind]
            if not krows:
                continue
            f = FIELDS[kind]
            out.append("| Case | " + " | ".join(h for _, h, _ in f) + " |")
            out.append("|---|" + "---|" * len(f))
            for label in dict.fromkeys(r["label"] for r in krows):
                g = [r for r in krows if r["label"] == label]
                out.append(f"| {label} | " + " | ".join(_cell([r[k] for r in g], fmt) for k, _, fmt in f) + " |")
            out.append("")
    return "\n".join(out)


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(suite, kind, key, label_has=""):
        g = [r for r in rows if r["suite"] == suite and r["kind"] == kind and label_has in r["label"]]
        xs = sorted({r["x"] for r in g})
        ys = [np.nanmean([r[key] for r in g if r["x"] == x]) for x in xs]
        return xs, ys

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs, ys = series("scaling", "rescue", "found")
    axes[0].plot(xs, ys, "o-", label="victims found (of 8)")
    xs, ys = series("scaling", "nav", "t90")
    ax2 = axes[0].twinx()
    ax2.plot(xs, ys, "s--", color="tab:orange", label="forest: 90% arrived (s)")
    axes[0].set_xlabel("drones"); axes[0].set_title("Scaling"); axes[0].legend(loc="lower left")
    ax2.legend(loc="lower right")
    xs, ys = series("radio", "rescue", "found", "packet loss")
    axes[1].plot([x * 100 for x in xs], ys, "o-", label="rescue: victims found")
    xs, ys = series("radio", "formation", "slot_error")
    ax3 = axes[1].twinx()
    ax3.plot([x * 100 for x in xs], ys, "s--", color="tab:orange", label="formation: slot error (m)")
    axes[1].set_xlabel("packet loss (%)"); axes[1].set_title("Radio degradation")
    axes[1].legend(loc="lower left"); ax3.legend(loc="upper right")
    xs, ys = series("sensors", "nav", "obstacle_hit_steps", "noise")
    axes[2].plot(xs, ys, "o-", label="obstacle hits")
    xs2, ys2 = series("sensors", "nav", "collision_steps", "noise")
    axes[2].plot(xs2, ys2, "s--", label="drone collisions")
    axes[2].set_xlabel("lidar noise std (m)"); axes[2].set_title("Sensor noise (forest)"); axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", default="scaling,radio,failures,sensors")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--out", default="outputs/benchmarks")
    args = p.parse_args(argv)
    suites = set(args.suites.split(","))
    jobs = [(c, s) for c in cases() if c["suite"] in suites for s in range(1, args.seeds + 1)]
    # longest jobs first so the pool finishes evenly
    jobs.sort(key=lambda j: -(j[0]["steps"] * j[0]["drones"]))
    print(f"Running {len(jobs)} simulations on {args.workers} workers ...", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, row in enumerate(ex.map(run_one, jobs), 1):
            rows.append(row)
            print(f"  [{k}/{len(jobs)}] {row['label']:45s} seed {row['seed']}: coll={row['collision_steps']} "
                  f"hits={row['obstacle_hit_steps']} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out / "robustness.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    table = summarise(rows)
    (out / "robustness.md").write_text(f"# Robustness experiments\n\n{args.seeds} seeds per row.\n" + table + "\n",
                                       encoding="utf-8")
    if suites == {"scaling", "radio", "failures", "sensors"}:
        plot(rows, out)
    print(table)
    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
