# Implementation Roadmap

**Decisions:** 2D world · point-mass dynamics · Python (no Unity: the
development laptop has an i5-8365U, Intel UHD 620 integrated graphics, no
CUDA GPU and limited disk) · every behaviour gets both a hand-written
controller and an RL policy, benchmarked against each other ·
work happens on the `development` branch.

All controllers implement `Controller.compute(state, ctx) -> accel (N, 2)`,
so a learned policy is a drop-in replacement for a hand-written one.

| # | Phase | Deliverable |
|---|---|---|
| 0 | Setup | Config, seeded fixed-dt engine, metrics, tests ✅ |
| 1 | Boids | Flocking with 20–50 drones, 0 collisions, Pygame viewer ✅ |
| 2 | Obstacles & avoidance | Circle/polygon obstacles, ray-cast range sensor, APF then ORCA safety layer |
| 3 | Communication | Range-limited messages, packet loss/latency, comm-graph view, shared occupancy map |
| 4 | Path planning | A* / Theta* on the occupancy grid, D* Lite replanning, path smoothing, maze scenario |
| 5 | Formation | Leader-follower + virtual structure, line/V/circle switching, squeezing through gaps |
| 6 | Search & rescue | Voronoi / auction area split, frontier exploration, PSO on victim signal, CBBA task allocation |
| 7 | Terrain & threats | Heightmap cost, threat zones, PSO multi-UAV path planner |
| 8 | Multi-agent RL | PettingZoo env; shared-parameter MAPPO (CTDE) per behaviour: avoidance, flocking, formation, search; curriculum learning |
| 9 | Experiments | Scaling 20→50, comm loss, drone failure, RL vs classical, demo video |

## RL on this hardware

There's no CUDA GPU, so training runs on 8 CPU threads. To stay practical:
- small MLP / GNN policies (< 100k parameters), shared across drones
- vectorised NumPy environment (no rendering during training)
- curriculum: train avoidance with 5–10 drones first, then scale up
- one behaviour at a time; warm-start from the previous checkpoint
- free Google Colab / Kaggle GPU if any run takes too long

## Research reading list

- Reynolds 1987 *Flocks, Herds and Schools*; Reynolds 1999 *Steering Behaviors*
- Olfati-Saber 2006, flocking with obstacles
- van den Berg et al. 2011, ORCA (*Reciprocal n-Body Collision Avoidance*)
- Koenig & Likhachev 2002, D* Lite; Theta*; Jump Point Search
- Ren & Beard, consensus-based formation control
- Kennedy & Eberhart 1995, PSO; PSO-based multi-UAV path planning
- Yamauchi 1997, frontier exploration; Cortés et al. 2004, Voronoi coverage
- Choi et al. 2009, CBBA task allocation
- Yu et al. 2021, MAPPO; PettingZoo docs
