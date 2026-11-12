"""Record animated GIFs of every major behaviour (the project demo).

Runs the real viewer offscreen (no window needed) and saves frames with
Pillow, so it works on any machine, including headless ones.

  python -m experiments.make_demo                    # all demos -> outputs/demo/*.gif
  python -m experiments.make_demo --only flocking,rescue
  python -m experiments.make_demo --scale 0.5 --colors 128 --stride 2 --fps 6 --out docs/demo   # README copies

Small markers (victims, slot squares) are only 1-2 px wide at half size and
blend into the background; use the full-size GIFs to see every detail.

Each GIF shows the arena and the live statistics panel.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# name: (title, sim overrides, simulated seconds, record every N steps, viewer settings)
DEMOS = {
    "01_flocking": ("Boids flocking, 30 drones", {"drones": {"count": 30}}, 60, 8, {"link_mode": "sensing"}),
    "02_avoidance": ("Pillar forest with ORCA, 40 drones", {"drones": {"count": 40}, "scenario": {"name": "forest"}},
                     60, 8, {"link_mode": "off"}),
    "03_shared_mapping": ("Radio + multi-hop map sharing, 40 drones",
                          {"drones": {"count": 40}, "scenario": {"name": "survey"}}, 40, 6,
                          {"map_mode": "team", "link_mode": "radio"}),
    "04_maze_planning": ("Maze: each drone plans on its own map, 25 drones",
                         {"drones": {"count": 25}, "scenario": {"name": "maze"}, "planning": {"enabled": True}},
                         130, 16, {"map_mode": "drone", "link_mode": "off"}),
    "05_formation_gap": ("V formation squeezes through a gap, 20 drones",
                         {"drones": {"count": 20}, "scenario": {"name": "corridor"},
                          "formation": {"enabled": True, "shape": "v"}}, 90, 10, {"link_mode": "off"}),
    "06_leader_failure": ("Leader + 4 drones fail, a new leader takes over",
                          {"drones": {"count": 16}, "formation": {"enabled": True, "shape": "v"},
                           "faults": {"fail_time": 15.0, "fail_fraction": 0.25, "fail_leader": True}},
                          50, 6, {"link_mode": "off"}),
    "07_search_rescue": ("Search and rescue: 20 drones, 8 hidden victims",
                         {"drones": {"count": 20}, "scenario": {"name": "rescue"}}, 240, 24,
                         {"map_mode": "team", "link_mode": "off"}),
    "08_terrain_routes": ("Hybrid PSO routes over terrain and threat zones, 12 drones",
                          {"drones": {"count": 12}, "scenario": {"name": "terrain"}}, 70, 8, {"link_mode": "off"}),
}


def record(name: str, out_dir: Path, scale: float, fps: int, colors: int = 256, stride: int = 1) -> Path:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from PIL import Image

    from swarm.config import load_config
    from swarm.sim.engine import Simulation
    from swarm.viz.render2d import Viewer

    title, over, seconds, every, view = DEMOS[name]
    sim = Simulation(load_config(None, over))
    v = Viewer(sim)
    for k, val in view.items():
        setattr(v, k, val)
    font = pygame.font.SysFont("consolas", 20, bold=True)
    frames = []
    steps = int(seconds / sim.cfg.sim.dt)
    every *= stride
    for s in range(0, steps, every):
        v._advance(every)
        v.screen.fill((16, 20, 28))
        v._draw_world()
        v._draw_swarm()
        v._draw_hud()
        v.screen.blit(font.render(title, True, (235, 235, 245)), (24, 18))
        raw = pygame.image.tobytes(v.screen, "RGB")
        img = Image.frombytes("RGB", v.screen.get_size(), raw)
        if scale != 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        frames.append(img)
    # one shared palette built from frames across the WHOLE run (colours like found
    # victims or mapped obstacles only appear later), no dithering: consistent colours
    # and far better GIF compression than a fresh, dithered palette per frame
    sample = frames[:: max(1, len(frames) // 8)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, f in enumerate(sample):
        strip.paste(f, (0, i * f.height))
    palette = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path = out_dir / f"{name}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0,
                   optimize=True)
    pygame.quit()
    return path


def _job(args):
    name, out, scale, fps, colors, stride = args
    t0 = time.perf_counter()
    path = record(name, Path(out), scale, fps, colors, stride)
    return name, path, time.perf_counter() - t0


def main(argv=None) -> None:
    from concurrent.futures import ProcessPoolExecutor
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="comma-separated demo names (prefix match), e.g. flocking,rescue")
    p.add_argument("--scale", type=float, default=0.7, help="frame size relative to the 1200x760 viewer")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--colors", type=int, default=256, help="palette size (64 keeps README copies small)")
    p.add_argument("--stride", type=int, default=1, help="keep every n-th frame (2 halves the size)")
    p.add_argument("--out", default="outputs/demo")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = p.parse_args(argv)
    names = list(DEMOS)
    if args.only:
        wanted = args.only.split(",")
        names = [n for n in names if any(w in n for w in wanted)]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(min(args.workers, len(names))) as ex:
        jobs = [(n, args.out, args.scale, args.fps, args.colors, args.stride) for n in names]
        for name, path, secs in ex.map(_job, jobs):
            print(f"  {name:22s} -> {path}  ({path.stat().st_size / 1e6:.1f} MB, {secs:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
