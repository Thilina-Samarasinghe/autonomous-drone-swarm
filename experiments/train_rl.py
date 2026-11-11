"""Train shared policies for the swarm behaviours (CPU, in parallel processes).

  python -m experiments.train_rl                                 # all 4 tasks
  python -m experiments.train_rl --tasks navigate --samples 2000000
  python -m experiments.train_rl --tasks flock,formation --drones 8

Each task runs in its own process with one PyTorch thread (4 tasks on a
4-core laptop). Writes models/<task>.pt (the policy, a few hundred KB),
models/<task>_log.json and models/<task>_curve.png.

Curriculum: navigate first trains with 4 drones, then the full number.
`--init models --drones 20` continues every policy with a bigger swarm
(the policies trained with 8 drones fell apart at 20).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DEFAULT_SAMPLES = {"navigate": 2_000_000, "flock": 1_200_000, "formation": 1_000_000, "search": 1_000_000}


def train_task(job: tuple[str, int, int, int, str, str | None]) -> dict:
    task, samples, drones, seed, out, init = job
    import torch
    torch.set_num_threads(1)
    from swarm.rl.env import SwarmEnv
    from swarm.rl.ppo import load_policy, save_policy, train

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    history = []
    actor = None
    prior = {}
    if init:
        # continue training an existing policy (curriculum: e.g. 8 -> 20 drones)
        actor, prior = load_policy(f"{init}/{task}.pt")
        old = Path(f"{init}/{task}_log.json")
        if old.exists():
            history = json.loads(old.read_text())["log"]
        stages = [(drones, samples)]
    else:
        stages = [(4, samples // 4), (drones, samples - samples // 4)] if task == "navigate" else [(drones, samples)]
    for n, budget in stages:
        env = SwarmEnv(task, n=n, seed=seed + n, shared_goal_prob=0.5 if (task == "navigate" and init) else 0.0)
        print(f"[{task}] training with {n} drones for {budget} samples", flush=True)
        actor, log = train(env, budget, seed=seed, actor=actor, log_every=25)
        offset = history[-1]["samples"] if history else 0
        for e in log.updates:
            e["drones"] = n
            e["samples"] += offset
        history += log.updates
    stages_done = prior.get("stages", [{"drones": prior.get("drones"), "samples": prior.get("samples")}]) if init else []
    meta = {"task": task, "drones": drones, "samples": samples, "seed": seed,
            "train_seconds": time.perf_counter() - t0 + prior.get("train_seconds", 0.0),
            "stages": stages_done + [{"drones": drones, "samples": samples}]}
    save_policy(actor, str(out_dir / f"{task}.pt"), meta)
    (out_dir / f"{task}_log.json").write_text(json.dumps({"meta": meta, "log": history}, indent=1))
    _plot(task, history, out_dir / f"{task}_curve.png")
    return meta


def _plot(task, history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    x = [h["samples"] for h in history]
    keys = [k for k in ("arrivals", "alignment", "slot_error", "coverage", "collisions") if k in history[-1]]
    fig, axes = plt.subplots(1, 1 + len(keys), figsize=(4 * (1 + len(keys)), 3.2))
    axes = np.atleast_1d(axes)
    axes[0].plot(x, [h["return"] for h in history])
    axes[0].set_title(f"{task}: episode return")
    for ax, k in zip(axes[1:], keys):
        ax.plot(x, [h.get(k, np.nan) for h in history])
        ax.set_title(k)
    for ax in axes:
        ax.set_xlabel("training samples")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="navigate,flock,formation,search")
    p.add_argument("--samples", type=int, help="override the per-task sample budget")
    p.add_argument("--drones", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--init", help="continue from the policies in this folder (curriculum stage)")
    args = p.parse_args(argv)
    tasks = args.tasks.split(",")
    jobs = [(t, args.samples or DEFAULT_SAMPLES[t], args.drones, args.seed, args.out, args.init) for t in tasks]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(min(args.workers, len(jobs))) as ex:
        for meta in ex.map(train_task, jobs):
            print(f"done: {meta['task']} in {meta['train_seconds'] / 60:.1f} min", flush=True)
    print(f"All done in {(time.perf_counter() - t0) / 60:.1f} min. Policies in {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
