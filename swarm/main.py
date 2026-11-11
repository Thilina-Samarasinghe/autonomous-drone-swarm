"""Entry point.

  python -m swarm.main                       # interactive viewer
  python -m swarm.main --drones 50           # override drone count
  python -m swarm.main --scenario forest --avoidance hybrid
  python -m swarm.main --headless --steps 3000 --out outputs/run1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from swarm.config import load_config
from swarm.control.factory import METHODS
from swarm.sim.scenarios import SCENARIOS
from swarm.sim.engine import Recorder, Simulation

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Autonomous drone swarm simulator")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--drones", type=int, help="number of drones")
    p.add_argument("--seed", type=int)
    p.add_argument("--scenario", choices=SCENARIOS, help="world layout")
    p.add_argument("--avoidance", choices=METHODS, help="obstacle / collision avoidance method")
    p.add_argument("--headless", action="store_true", help="run without a window")
    p.add_argument("--steps", type=int, help="steps for a headless run")
    p.add_argument("--out", default="outputs/latest", help="output folder for headless runs")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    overrides: dict = {}
    if args.drones is not None:
        overrides.setdefault("drones", {})["count"] = args.drones
    if args.seed is not None:
        overrides.setdefault("sim", {})["seed"] = args.seed
    if args.steps is not None:
        overrides.setdefault("sim", {})["steps"] = args.steps
    if args.scenario is not None:
        overrides.setdefault("scenario", {})["name"] = args.scenario
    if args.avoidance is not None:
        overrides.setdefault("avoidance", {})["method"] = args.avoidance
    cfg = load_config(args.config, overrides)
    sim = Simulation(cfg)

    if not args.headless:
        from swarm.viz.render2d import Viewer
        Viewer(sim).run()
        return

    from swarm.viz.plots import plot_metrics, plot_trajectories

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(every=5)
    t0 = time.perf_counter()
    history = sim.run(cfg.sim.steps, rec)
    wall = time.perf_counter() - t0

    import numpy as np
    positions = np.stack(rec.positions)
    plot_trajectories(positions, sim.world, str(out / "trajectories.png"), sim.goal,
                      cfg.scenario.goal_radius,
                      f"{cfg.scenario.name} / {cfg.avoidance.method} - {cfg.drones.count} drones")
    plot_metrics(history, cfg.drones.radius, str(out / "metrics.png"))

    tail = history[len(history) // 2:]
    summary = {
        "scenario": cfg.scenario.name,
        "avoidance": cfg.avoidance.method,
        "drones": cfg.drones.count,
        "steps": cfg.sim.steps,
        "sim_time_s": round(sim.time, 2),
        "wall_time_s": round(wall, 2),
        "steps_per_second": round(cfg.sim.steps / wall, 1),
        "total_collision_steps": sim.total_collisions,
        "min_separation_m": round(min(m.min_separation for m in history), 3),
        "mean_polarization_2nd_half": round(sum(m.polarization for m in tail) / len(tail), 3),
        "mean_local_polarization_2nd_half": round(sum(m.local_polarization for m in tail) / len(tail), 3),
        "final_groups": history[-1].groups,
        "total_obstacle_hits": sim.total_obstacle_hits,
        "min_obstacle_clearance_m": (round(min(m.min_clearance for m in history), 3)
                                     if sim.world.obstacles.count else None),
        "arrived_final": history[-1].arrived if sim.goal is not None else None,
        "time_to_90pct_arrived_s": (round(sim.time_to_arrive_90, 2)
                                    if sim.time_to_arrive_90 is not None else None),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Plots written to {out.resolve()}")


if __name__ == "__main__":
    main()
