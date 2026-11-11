"""Entry point.

  python -m swarm.main                       # interactive viewer
  python -m swarm.main --drones 50           # override drone count
  python -m swarm.main --scenario forest --avoidance hybrid
  python -m swarm.main --scenario survey --packet-loss 0.3
  python -m swarm.main --headless --scenario survey --no-share --out outputs/noshare
  python -m swarm.main --scenario maze --plan
  python -m swarm.main --scenario corridor --formation v
  python -m swarm.main --headless --steps 3000 --out outputs/run1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from swarm.config import load_config
from swarm.control.factory import METHODS
from swarm.formation.shapes import SHAPES
from swarm.planning.planner import ALGORITHMS
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
    p.add_argument("--packet-loss", type=float, help="radio packet loss probability 0..1")
    p.add_argument("--comm-range", type=float, help="radio range in metres")
    p.add_argument("--no-share", action="store_true", help="drones do not share maps")
    p.add_argument("--no-relay", action="store_true", help="single-hop radio (no forwarding)")
    p.add_argument("--formation", choices=SHAPES, help="fly in formation with this shape")
    p.add_argument("--no-squeeze", action="store_true", help="formation never collapses to a column for gaps")
    p.add_argument("--plan", action="store_true", help="enable path planning on each drone's map")
    p.add_argument("--planner", choices=ALGORITHMS, help="planning algorithm (implies --plan)")
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
    comms = {}
    if args.packet_loss is not None:
        if not 0.0 <= args.packet_loss <= 1.0:
            raise SystemExit("--packet-loss must be between 0 and 1")
        comms["packet_loss"] = args.packet_loss
    if args.comm_range is not None:
        comms["range"] = args.comm_range
    if args.no_share:
        comms["share_maps"] = False
    if args.no_relay:
        comms["relay"] = False
    if comms:
        overrides["comms"] = comms
    if args.formation:
        overrides["formation"] = {"enabled": True, "shape": args.formation}
        if args.no_squeeze:
            overrides["formation"]["adaptive"] = False
    if args.plan or args.planner:
        overrides["planning"] = {"enabled": True}
        if args.planner:
            overrides["planning"]["algorithm"] = args.planner
    cfg = load_config(args.config, overrides)
    sim = Simulation(cfg)

    if not args.headless:
        from swarm.viz.render2d import Viewer
        Viewer(sim).run()
        return

    from swarm.viz.plots import plot_info, plot_metrics, plot_trajectories

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(every=5)
    info_every = max(1, round(1.0 / cfg.sim.dt))          # sample knowledge once per second
    history, info = [], [sim.info_metrics()]
    t0 = time.perf_counter()
    for _ in range(cfg.sim.steps):
        history.append(sim.step())
        rec.capture(sim)
        if sim.step_count % info_every == 0:
            info.append(sim.info_metrics())
    wall = time.perf_counter() - t0

    import numpy as np
    positions = np.stack(rec.positions)
    plot_trajectories(positions, sim.world, str(out / "trajectories.png"), sim.goal,
                      cfg.scenario.goal_radius,
                      f"{cfg.scenario.name} / {cfg.avoidance.method} - {cfg.drones.count} drones")
    plot_metrics(history, cfg.drones.radius, str(out / "metrics.png"))
    if sim.maps is not None or sim.comms is not None:
        plot_info(info, str(out / "comms_mapping.png"))

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
    last = info[-1]
    summary.update({
        "map_sharing": cfg.comms.share_maps,
        "packet_loss": cfg.comms.packet_loss,
        "relay": cfg.comms.relay,
        "map_coverage_mean_final": round(last.map_coverage_mean, 3),
        "map_coverage_min_final": round(last.map_coverage_min, 3),
        "map_coverage_team_final": round(last.map_coverage_team, 3),
        "map_accuracy_final": round(last.map_accuracy, 4),
        "time_to_50pct_mean_coverage_s": next(
            (round(i.time, 1) for i in info if i.map_coverage_mean >= 0.5), None),
        "messages_delivered": last.delivered,
        "messages_dropped": last.dropped,
        "kbytes_sent": round(last.kbytes_sent, 1),
    })
    if sim.formation is not None:
        fs = sim.formation.stats
        summary.update({
            "formation": cfg.formation.shape,
            "formation_error_mean_m": round(fs.mean_error, 3),
            "formation_in_shape_pct": round(100 * fs.in_shape_fraction, 1),
            "formation_squeezes": fs.squeezes,
            "leader_arrived_s": (round(sim.time_leader_arrived, 2)
                                 if sim.time_leader_arrived is not None else None),
        })
    if sim.planner is not None:
        ps = sim.planner.stats
        summary.update({
            "planning": cfg.planning.algorithm,
            "plans": ps.plans,
            "plan_ms_mean": round(ps.mean_ms, 2),
            "blocked_replans": ps.blocked_replans,
            "failed_plans": ps.failed,
        })
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Plots written to {out.resolve()}")


if __name__ == "__main__":
    main()
