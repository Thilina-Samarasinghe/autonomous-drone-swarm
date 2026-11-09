# Autonomous Drone Swarm Simulation

A 2D, decentralised drone-swarm simulator. Every drone decides for itself using
only what it can sense within its perception radius; cooperation emerges from
local rules (and, later, learned policies).

**Stack:** Python 3.12 · NumPy · SciPy · Pygame-CE · Matplotlib · PyTorch (CPU, Phase 8)

## Status

| Phase | Feature | Status |
|---|---|---|
| 0 | Project setup, config, fixed-step seeded engine, metrics | ✅ done |
| 1 | Boids flocking (separation / alignment / cohesion) + 2D viewer | ✅ done |
| 2 | Obstacles, range sensors, ORCA collision avoidance | ⏳ next |
| 3 | Communication network + shared map | |
| 4 | A* / D* Lite path planning | |
| 5 | Formation flying | |
| 6 | Cooperative search & rescue (PSO, coverage, task allocation) | |
| 7 | Terrain + threat zones, PSO multi-UAV planner | |
| 8 | Multi-agent RL (MAPPO) for every behaviour | |
| 9 | Benchmarks, robustness experiments, demo | |

See [docs/PLAN.md](docs/PLAN.md) for the full roadmap.

## Quick start

```bash
pip install -r requirements.txt

python -m swarm.main                     # interactive viewer (30 drones)
python -m swarm.main --drones 50         # 50 drones
python -m swarm.main --headless --steps 3000 --drones 50 --out outputs/run1
python -m pytest                         # tests
```

A headless run writes `trajectories.png`, `metrics.png` and `summary.json`.

### Viewer controls

| Key | Action |
|---|---|
| `SPACE` / `.` | pause / single-step |
| Left click / right click | set / clear a shared goal |
| `L` `C` `T` | toggle neighbour links / perception circles / trails |
| `+` / `-` | add / remove 5 drones |
| `Q/A` `W/S` `E/D` | separation / alignment / cohesion weight ±0.1 |
| `[` / `]` | slower / faster simulation |
| `R` `H` `ESC` | reset / help / quit |

## How Phase 1 works

Point-mass drones (max speed 8 m/s, max acceleration 6 m/s²) run Reynolds
boids with three improvements that made the difference between a flock that
collides and one that doesn't:

1. **Prioritised acceleration**: separation and wall avoidance take the
   acceleration budget first; alignment and cohesion only get what's left,
   so they can never cancel collision avoidance.
2. **Anticipatory separation**: each drone predicts neighbour positions
   0.6 s ahead from relative velocity and reacts to the closer of now vs.
   predicted. This removed the pile-ups that happened when a flock turned at a wall.
3. **Urgency-scaled separation**: gentle at the personal-space edge, full
   braking near contact.

Neighbour search uses a KD-tree, so cost grows ~N log N.

### Phase 1 benchmark (3000 steps = 150 s simulated, 5 seeds each)

| Drones | Collisions | Closest approach | Local alignment | Speed |
|---|---|---|---|---|
| 20 | 0 | ≥ 1.57 m | 0.97–0.99 | ~680 steps/s |
| 30 | 0 | ≥ 1.51 m | 0.97–0.98 | ~630 steps/s |
| 50 | 0 | ≥ 1.55 m | 0.97–0.98 | ~500 steps/s |

Collision distance is 1.0 m (two 0.5 m bodies). 500 steps/s is about 25×
real time on a laptop i5-8365U with no GPU.

## Layout

```
configs/default.yaml     all tunable parameters
swarm/config.py          typed config loader
swarm/agents/state.py    swarm state (positions, velocities, ids)
swarm/sim/               world, dynamics, neighbour search, engine
swarm/control/           controller interface + boids
swarm/metrics.py         separation, collisions, polarization, groups
swarm/viz/               pygame viewer, matplotlib plots
tests/                   pytest suite
```
