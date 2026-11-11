# Autonomous Drone Swarm Simulation

> 🚧 **Work in progress.** Phases 0–4 (simulation core, boids flocking,
> obstacle avoidance with lidar + ORCA, radio communication with shared
> mapping, path planning on each drone's own map) are built. Formation flying,
> search-and-rescue and reinforcement learning are **still being built**.
> See [Roadmap](#roadmap) below.

A 2D simulator for a swarm of **20–50 autonomous drones**. Every drone makes its
own decisions using only what it can sense around itself, and cooperative
behaviour such as flocking, formations and shared search emerges from those
local decisions. The project shows **distributed decision-making and collective
intelligence**, not one central AI controlling every drone.

---

## Table of contents

1. [Project goals](#project-goals)
2. [Current status](#current-status)
3. [Features](#features)
4. [Quick start](#quick-start)
5. [Usage](#usage)
6. [Viewer controls](#viewer-controls)
7. [Configuration](#configuration)
8. [How it works](#how-it-works)
9. [Metrics](#metrics)
10. [Benchmark results](#benchmark-results)
11. [Project structure](#project-structure)
12. [Testing](#testing)
13. [Roadmap](#roadmap)
14. [Algorithms (planned and implemented)](#algorithms)
15. [Hardware and design decisions](#hardware-and-design-decisions)
16. [Development workflow](#development-workflow)
17. [References](#references)

---

## Project goals

| Goal | Description |
|---|---|
| Autonomous obstacle avoidance | Drones sense and steer around static and dynamic obstacles |
| Collision prevention | No drone-to-drone collisions, even with 50 drones in tight spaces |
| Formation flying | Line, V and circle formations that can reshape to pass through gaps |
| Dynamic path planning | Plan routes on a map and replan when new obstacles are discovered |
| Cooperative search and rescue | Split an area between drones, search it, find victims, report and converge |
| Communication | Drones exchange information with limited range, packet loss and latency |
| Learning | Every behaviour gets a hand-written controller **and** a reinforcement-learning policy, compared head to head |

---

## Current status

| Phase | Feature | Status |
|---|---|---|
| 0 | Project setup: config system, seeded fixed-timestep engine, metrics, tests | ✅ Done |
| 1 | Boids flocking + real-time 2D viewer + headless runner with plots | ✅ Done |
| 2 | Obstacles, lidar sensing, APF + ORCA collision avoidance, scenarios, benchmark | ✅ Done |
| 3 | Radio network, per-drone occupancy maps, multi-hop map sharing, benchmark | ✅ Done |
| 4 | Path planning (A* and cost field) on each drone's own map, replanning, maze, benchmark | ✅ Done (under review) |
| 5 | Formation flying | 🔜 Next |
| 6 | Cooperative search and rescue (PSO, coverage, task allocation) | 📋 Planned |
| 7 | Terrain cost and threat zones, PSO multi-drone path planner | 📋 Planned |
| 8 | Multi-agent reinforcement learning (MAPPO) for **all** behaviours | 📋 Planned |
| 9 | Benchmarks, robustness experiments, demo video | 📋 Planned |

---

## Features

### ✅ Implemented

- **Decentralised point-mass drones**: max speed 8 m/s, max acceleration 6 m/s².
  Each drone reacts only to neighbours inside its 15 m perception radius.
- **Boids flocking** (separation, alignment, cohesion) with three improvements:
  - *prioritised acceleration*: collision avoidance always wins over flocking
  - *anticipatory separation*: predicts neighbours 0.6 s ahead
  - *urgency-scaled separation*: gentle far away, full braking near contact
- **Wall avoidance**: soft steering near the arena edge plus hard walls.
- **Shared goal**: click anywhere in the viewer and the swarm flies there.
- **Obstacles** (Phase 2): circles and boxes with exact signed-distance and
  ray-intersection geometry, plus hard physics (drones can't pass through).
- **Onboard lidar** (Phase 2): 36 beams, 20 m range. Drones know about
  obstacles *only* through their own scan; the nearest point of each
  obstacle is extracted from the scan's local minima.
- **Four avoidance methods**, switchable live (Phase 2):
  - `none`: flocking only (baseline)
  - `apf`: artificial potential field from lidar
  - `orca`: Optimal Reciprocal Collision Avoidance safety layer (RVO2 solver port), **default**
  - `hybrid`: APF steering + ORCA safety layer
- **Scenarios** (Phase 2): `open`, `forest` (pillar grid), `corridor`
  (wall with one 14 m gap), `random` (seeded mix of circles and boxes).
- **Avoidance benchmark** (Phase 2): every method × scenario × seed in parallel,
  writing a CSV and a markdown results table.
- **Occupancy-grid map per drone** (Phase 3): 2 m cells, log-odds updates
  from the drone's own lidar, scored against ground truth (99.8%+ accurate).
- **Radio network** (Phase 3): 30 m range, packet loss, latency, a broadcast
  rate and a per-message bandwidth cap. Each drone keeps a beacon table of
  where it believes the others are.
- **Multi-hop map sharing** (Phase 3): drones broadcast newly learned map
  cells plus a random sample of known ones (anti-entropy, so lost packets
  are eventually repaired). Receivers keep the more confident value, so
  information relayed over many hops is never double-counted. Sharing and
  relaying can each be switched off for comparison.
- **`survey` scenario** (Phase 3): obstacles across the whole arena, drones
  start scattered and roam in several flocks, no goal.
- **Communication benchmark** (Phase 3): no sharing vs single-hop vs
  multi-hop at 0/20/50/80% packet loss.
- **Path planning on each drone's own map** (Phase 4): a cost map (blocked /
  inflated / near-obstacle / unknown cells) and two planners:
  - **A\*** with the octile heuristic (pure Python, the textbook algorithm)
  - **cost field** (default): a backward Dijkstra search from the goal (SciPy, in C), the
    quantity D\* Lite maintains. One search gives the best route from *every*
    cell, so a drone that gets pushed off its path re-attaches with a lookup.
    It finds paths exactly as short as A\* (verified by tests) and is about 5× faster.
- **Optimistic replanning** (Phase 4): unknown space is assumed passable;
  drones replan when a cell on their path turns out to be a wall, when their
  map's obstacle layout changes, and every 5 s.
- **Smooth path following** (Phase 4): each drone steers at the farthest point
  on its path it can see within 12 m (online string pulling). ORCA stays on
  as the safety layer.
- **`maze` scenario** (Phase 4): an S-shaped maze with a dead-end trap where flying straight at the goal fails.
- **Planning benchmark** (Phase 4): no planning vs own-map planning vs
  shared-map planning vs A\*.
- **Fast neighbour search**: KD-tree, roughly N log N cost.
- **Real-time Pygame viewer**: neighbour links, trails, perception circles,
  live stats panel, behaviour weights tunable while running, add or remove drones.
- **Headless mode**: runs without a window and saves trajectory and metric plots
  plus a JSON summary.
- **Reproducible**: the same seed always gives the same run.
- **Test suite**: 15 pytest tests.

### 🚧 Still being built / planned

- Moving (dynamic) obstacles
- Incremental D\* Lite (re-using the previous search instead of a full recompute)
- Leader-follower and virtual-structure formations with shape switching
- Search and rescue: Voronoi area split, frontier exploration, PSO victim search,
  CBBA/auction task allocation, battery and return-to-base
- Terrain heightmap and threat zones with a PSO multi-drone planner
- PettingZoo multi-agent environment and MAPPO training for every behaviour
- Experiment runner: scaling from 20 to 50 drones, communication loss, drone failures,
  RL vs. classical comparison

---

## Quick start

**Requirements:** Python 3.10+ (developed on 3.12). No GPU needed.

```bash
git clone <repo-url>
cd autonomous-drone-swarm
git checkout development

pip install -r requirements.txt
python -m swarm.main
```

On a machine with little disk space, install with `pip install --no-cache-dir -r requirements.txt`.

---

## Usage

```bash
# Interactive viewer (30 drones by default)
python -m swarm.main

# Choose the number of drones and the random seed
python -m swarm.main --drones 50 --seed 7

# Headless run: no window, writes plots and a summary
python -m swarm.main --headless --drones 50 --steps 3000 --out outputs/run1

# Obstacle scenarios and avoidance methods (Phase 2)
python -m swarm.main --scenario forest --drones 50
python -m swarm.main --scenario corridor --avoidance orca
python -m swarm.main --headless --scenario random --avoidance apf --out outputs/apf_random

# Communication and shared mapping (Phase 3)
python -m swarm.main --scenario survey --drones 50
python -m swarm.main --scenario survey --packet-loss 0.5
python -m swarm.main --headless --scenario survey --no-share --out outputs/noshare
python -m swarm.main --headless --scenario survey --no-relay --out outputs/singlehop

# Path planning (Phase 4)
python -m swarm.main --scenario maze --plan
python -m swarm.main --scenario maze --planner astar
python -m swarm.main --headless --scenario maze --plan --no-share --out outputs/maze_ownmap

# Benchmark planning settings in the maze (writes outputs/benchmarks/planning.*)
python -m experiments.benchmark_planning                        # 30 drones x 3 seeds

# Benchmark communication settings (writes outputs/benchmarks/comms.*)
python -m experiments.benchmark_comms                           # 50 drones x 3 seeds

# Benchmark every avoidance method on every scenario (writes outputs/benchmarks/)
python -m experiments.benchmark_avoidance                       # 50 drones x 5 seeds
python -m experiments.benchmark_avoidance --drones 30 --seeds 3 --steps 1500   # quicker

# Use your own config file
python -m swarm.main --config configs/my_experiment.yaml
```

| Option | Default | Meaning |
|---|---|---|
| `--config` | `configs/default.yaml` | YAML configuration file |
| `--drones` | from config (30) | Number of drones |
| `--seed` | from config (42) | Random seed |
| `--scenario` | from config (`open`) | `open`, `forest`, `corridor`, `random`, `survey`, `maze` |
| `--avoidance` | from config (`orca`) | `none`, `apf`, `orca`, `hybrid` |
| `--packet-loss` | from config (0) | Radio packet loss probability, 0–1 |
| `--comm-range` | from config (30) | Radio range in metres |
| `--no-share` | off | Drones don't share maps |
| `--no-relay` | off | Single-hop radio: no forwarding of others' information |
| `--plan` | off | Each drone plans routes on its own map |
| `--planner` | from config (`field`) | `field` or `astar` (implies `--plan`) |
| `--headless` | off | Run without a window |
| `--steps` | from config (3000) | Steps for a headless run (1 step = 0.05 s) |
| `--out` | `outputs/latest` | Output folder for headless results |

A headless run writes:

| File | Contents |
|---|---|
| `trajectories.png` | Every drone's path (■ start, ● end) |
| `metrics.png` | Min separation, polarization, mean speed and groups over time |
| `comms_mapping.png` | Map coverage (team / average / worst drone), accuracy, radio groups and info age over time |
| `summary.json` | Collisions, obstacle hits, clearance, arrivals, alignment, map coverage and accuracy, time to 50% coverage, messages delivered / dropped, traffic, plans / ms per plan / blocked replans, run speed |

---

## Viewer controls

| Key / mouse | Action |
|---|---|
| `SPACE` | Pause / resume |
| `.` | Single step (while paused) |
| `1` … `6` | Scenario: open / forest / corridor / random / survey / maze |
| `K` | Path planning on / off (yellow line = selected drone's path, dot = its steering target) |
| `M` | Cycle avoidance method: none → apf → orca → hybrid |
| `V` | Toggle lidar beams (red = hit) |
| Left click | Set a shared goal point |
| Right click | Clear the goal |
| `L` | Links: radio (red = dropped message) → sensing → off |
| `G` | Map overlay: selected drone's map → team map → off |
| `Tab` | Select next drone (white ghosts = where it *believes* the others are) |
| `N` | Map sharing on / off |
| `P` | Packet loss: 0 → 20 → 50 → 80% |
| `C` | Toggle perception circles |
| `T` | Toggle trails |
| `+` / `-` | Add / remove 5 drones |
| `Q` / `A` | Separation weight +0.1 / −0.1 |
| `W` / `S` | Alignment weight +0.1 / −0.1 |
| `E` / `D` | Cohesion weight +0.1 / −0.1 |
| `[` / `]` | Slower / faster simulation |
| `R` | Reset |
| `H` | Show help |
| `ESC` | Quit |

Drone colour shows speed: blue is slow, cyan is cruising, yellow is fast.
The dashed red circle is the goal area; the panel shows how many drones arrived.
In the map overlay, green cells are known free and orange cells known occupied.
The big circle around the selected drone is its radio range.

---

## Configuration

All parameters live in [`configs/default.yaml`](configs/default.yaml). Units are
metres and seconds. Unknown keys raise an error, so typos are caught early.

| Section | Key | Default | Meaning |
|---|---|---|---|
| `sim` | `dt` | 0.05 | Seconds per physics step |
| | `seed` | 42 | Random seed |
| | `steps` | 3000 | Headless run length |
| `world` | `width` × `height` | 200 × 120 | Arena size (m) |
| | `boundary_margin` | 12 | Distance from the wall where drones start turning back |
| `drones` | `count` | 30 | Number of drones |
| | `radius` | 0.5 | Body radius (m); contact happens at 1.0 m |
| | `max_speed` | 8.0 | m/s |
| | `cruise_speed` | 5.0 | Preferred flocking speed |
| | `max_accel` | 6.0 | m/s² |
| | `perception_radius` | 15 | Sensing range |
| | `separation_radius` | 4 | Personal space |
| | `spawn_spacing` | 3 | Minimum distance between drones at start |
| | `spawn_speed` | 0.5 | Drones start near hover |
| `boids` | `separation_weight` | 2.5 | |
| | `alignment_weight` | 1.0 | |
| | `cohesion_weight` | 0.8 | |
| | `boundary_weight` | 3.0 | |
| | `goal_weight` | 1.2 | |
| `sensor` | `num_rays` | 36 | Lidar beams over 360° |
| | `range` | 20 | Lidar range (m) |
| `avoidance` | `method` | orca | `none` / `apf` / `orca` / `hybrid` |
| | `apf_weight` | 1.5 | Potential-field strength |
| | `apf_influence` | 8 | Obstacles closer than this repel (m) |
| | `orca_time_horizon` | 2.0 | Look-ahead for drone–drone constraints (s) |
| | `orca_obstacle_horizon` | 1.5 | Look-ahead for obstacle constraints (s) |
| | `orca_neighbor_dist` | 15 | Drones considered by ORCA (m) |
| | `orca_max_neighbors` | 10 | Max drones per ORCA problem |
| | `safety_margin` | 0.3 | Extra clearance added to the body radius (m) |
| `scenario` | `name` | open | `open` / `forest` / `corridor` / `random` |
| | `goal_radius` | 15 | Drones inside count as arrived (m) |
| | `num_obstacles` | 25 | Random scenario only |
| `mapping` | `enabled` | true | Build occupancy maps |
| | `cell_size` | 2.0 | Grid cell size (m) |
| | `update_interval` | 0.1 | Seconds between lidar → map updates |
| | `l_free` / `l_occ` / `l_max` | −0.4 / 0.85 / 4.0 | Log-odds update and clamp |
| | `known_threshold` | 1.0 | \|log-odds\| above this = cell known |
| `comms` | `enabled` | true | Radio on |
| | `range` | 30 | Radio range (m) |
| | `packet_loss` | 0.0 | Probability each delivery is lost |
| | `latency` | 0.1 | Seconds from send to receive |
| | `broadcast_interval` | 0.25 | Seconds between broadcasts (4 Hz) |
| | `share_maps` | true | Include map cells in messages |
| | `relay` | true | Multi-hop forwarding |
| | `max_cells_per_message` | 300 | Bandwidth cap |
| `planning` | `enabled` | false | Plan routes on each drone's map |
| | `algorithm` | field | `field` (backward Dijkstra) or `astar` |
| | `replan_interval` | 1.0 | Min seconds between replans when the map changed |
| | `refresh_interval` | 5.0 | Full replan at least this often |
| | `unknown_cost` | 0.3 | Extra cost of unknown cells (optimistic planning) |
| | `inflation_cost` | 8.0 | Cost of cells within body radius of an obstacle |
| | `clearance_cost` / `clearance_cells` | 2.0 / 2.0 | Extra cost near obstacles (keeps paths centred) |
| | `lookahead` | 12 | Metres along the path to look for the steering target |
| `render` | `window_width` × `window_height` | 1200 × 760 | |
| | `fps` | 60 | |
| | `trail_length` | 40 | Frames of trail per drone |
| | `steps_per_frame` | 1 | Simulation speed-up |

---

## How it works

### Simulation loop

```
each step (dt = 0.05 s):
  1. neighbours  = KD-tree query within perception radius
  2. lidar       = 36 beams per drone -> nearest obstacle points
  3. accel       = controller.compute(state, context)    # each drone: local info only
  4. dynamics    = clamp accel -> update velocity -> clamp speed -> update position
  5. physics     = walls + obstacles push drones out (counted as a hit)
  6. metrics     = separation, collisions, hits, clearance, arrivals, alignment
  7. mapping     = every 0.1 s each drone folds its lidar scan into its own map
  8. radio       = every 0.25 s each drone broadcasts; deliveries arrive after latency
  (planning, if on, runs before step 3 and turns the goal into per-drone waypoints)
```

### Controller interface

Every behaviour, hand-written or learned, implements one method:

```python
class Controller(Protocol):
    def compute(self, state: SwarmState, ctx: StepContext) -> np.ndarray:
        """Return desired acceleration for every drone, shape (N, 2)."""
```

So a trained RL policy is a drop-in replacement for the boids controller, and
the two can be benchmarked under identical conditions.

### Boids with prioritised acceleration

Rules are combined in priority order. Each rule may only use the acceleration
budget that is left after the rules above it (Reynolds 1999):

| Priority | Rule | Behaviour |
|---|---|---|
| 1 | Separation | Steer away from neighbours that are or *will be* (0.6 s ahead) inside personal space |
| 2 | Boundary | Turn back before hitting the arena wall |
| 3 | Goal | Fly to the clicked target, slowing down on arrival |
| 4 | Alignment + cohesion | Match neighbours' heading, drift toward the local centre |
| 5 | Cruise | Hold the preferred speed |

### Obstacle avoidance (Phase 2)

```
  boids (+ APF obstacle force)  --a_nominal-->  ORCA safety layer  --a_safe-->  dynamics
                                                      ^
              neighbours' positions & velocities -----+
              lidar nearest-obstacle points ----------+
```

**APF (artificial potential field).** Each obstacle point the lidar sees
within 8 m pushes the drone away. The push grows quadratically as the gap
shrinks, so distant pillars barely matter. A sideways component makes drones
slide *around* obstacles they're flying into instead of stopping dead. APF is
cheap and smooth but gives **no guarantee**: in a crowd it must trade drone
separation against obstacle clearance, and it can't do both.

**ORCA (Optimal Reciprocal Collision Avoidance).** Each drone turns the
boids output into a preferred velocity, then builds half-plane constraints
on its own next velocity:
- one per nearby drone: "don't collide within 2 s", with each drone taking
  half the effort (reciprocal), so no communication is needed
- one per lidar obstacle point: "approach no faster than gap / 1.5 s", with full
  effort because obstacles don't move

A small 2D linear program (a port of the RVO2 library solver) picks the
allowed velocity closest to the preferred one. If the constraints can't all
be met, it picks the velocity that violates them least.

**Hybrid.** APF steering with ORCA as the safety layer. It is just as safe
as ORCA, but the benchmark showed the extra APF push only slows the swarm
down (15–30% longer to reach the goal) without adding clearance, so **ORCA
alone is the default**.

### Communication and shared mapping (Phase 3)

**Occupancy grid.** Each drone's map is a grid of 2 m cells storing the
log-odds that the cell is occupied. A lidar beam passing through a cell
lowers it (free); a beam ending in a cell raises it (occupied). Values are
clamped so a cell can still change its mind. Drones never see the true map:
ground truth is used only to score them.

**Radio.** Every 0.25 s each drone broadcasts one message to every drone
within 30 m. Each delivery can be lost (`packet_loss`) and arrives after
`latency`. A message carries:
- a **beacon table**: where the sender believes every drone is, with timestamps;
  receivers keep whichever entry is newer
- up to 300 **map cells**: first the ones the sender recently became sure
  about, then a random sample of cells it already knows. The sample is
  *anti-entropy*: without it, a cell whose messages were all lost would never
  be sent again.

**Merging.** A receiver adopts a cell only when the sender is clearly more
confident about it. "More confident wins" can be applied any number of times
with the same result, so information that loops through the network is never
double-counted. Adopted cells are re-broadcast (relay), which is how
knowledge travels **multi-hop** between flocks that are out of direct range.
With `relay` off, drones forward nothing they learned from others.

### Path planning (Phase 4)

```
 own map (Phase 3) --> cost map --> planner (cost field or A*) --> path
                                                                   |
 drone position --> farthest visible point on path (<= 12 m) --> boids goal term --> ORCA --> dynamics
```

**Cost map.** Built from the drone's own log-odds map: known obstacles are
blocked; cells within the drone's body radius of an obstacle are expensive
(×8); cells close to obstacles cost a little extra (keeps paths centred in
corridors); unknown cells cost slightly more than known-free ones. Diagonal
moves may not cut between two blocked cells.

**Planners.**
- **A\*** searches from the drone to the goal with the octile-distance
  heuristic, which never overestimates, so the path is optimal. It's written in
  plain Python to show the algorithm, at ~35 ms per plan.
- **Cost field** (default) runs Dijkstra backwards from the goal using SciPy's
  C implementation, at ~7 ms per plan. The result is the cost-to-go from *every*
  cell. That's the quantity D\* Lite maintains, and it means a drone pushed off
  course just looks up the new best direction. Paths are exactly as short as
  A\*'s.

**Replanning.** The planner assumes unknown space is free ("optimistic").
When the drone, or a teammate via radio, discovers a wall, the map's
obstacle revision counter changes and the drone replans (at most once per
second, staggered across drones). A wall appearing *on* the current path
triggers an immediate replan.

**Path following.** The drone steers at the farthest waypoint within 12 m that it
can see in a straight line on its own map. This turns the staircase grid
path into smooth flight without a separate smoothing step.

Lessons from development (found by tests and benchmarks):

- Drones spawned at full speed in random directions collided before they
  could brake, so they now start near hover.
- A "desired minus current velocity" cohesion rule acted as a brake and
  halved flock speed, so it was replaced with a pure pull toward the centre.
- Flocks piled up when turning at walls. Anticipatory separation fixed this.
- *Phase 2:* drones stopped in front of the corridor gap. The cause was a
  raycast bug: a beam parallel to a box edge but outside the box was reported
  as a hit, so drones "saw" a wall inside the gap. Fixed and covered by a test.
- *Phase 2:* the first APF version used "steer away at full speed", which
  braked drones even when passing beside a pillar, so they never reached the
  goal. It was replaced with a true distance-based force.
- *Phase 2:* raycasting every beam against every obstacle took half the step
  time. Testing only obstacles within sensor range made it about 3× faster
  with identical results (verified against brute force).
- *Phase 3:* the first sharing design sent each new map cell once. If that
  message was lost, or nobody was in range, the cell was never shared again.
  Adding anti-entropy (a random sample of known cells per message) fixed it.
- *Phase 3:* in single-hop mode the anti-entropy sample could still forward
  cells learned from others, which is secretly multi-hop. Drones now track which
  cells they observed themselves, and single-hop sends only those.
- *Phase 3:* merging each delivered message separately was slow when a
  tight flock exchanged ~2,500 messages per broadcast. Messages are now
  grouped per receiver and merged in one vectorised pass.
- *Phase 4:* the first planner ran at 22 steps/s with 30 drones. Profiling
  showed the time spread across graph building, path extraction and steering
  point selection. Four fixes (replan only when the obstacle layout changed, a
  fixed neighbour table, vectorised path extraction, building the sparse
  graph directly) doubled it, with identical results.
- *Phase 4:* the first path-extraction code could step diagonally between two
  blocked cells, which the search itself forbids. Extraction now uses exactly
  the search's edges; a test checks every step of every path.

---

## Metrics

| Metric | Meaning | Good value |
|---|---|---|
| `min_separation` | Closest distance between any two drones (m) | > 1.0 (contact distance) |
| `collisions` | Pairs currently overlapping | 0 |
| `mean_speed` | Average speed (m/s) | Near cruise speed |
| `polarization` | Global heading alignment, 0 (random) to 1 (all the same) | High when there is one flock |
| `local_polarization` | How aligned each drone is with its own neighbours | > 0.9 for a healthy flock |
| `groups` | Connected components of the perception graph | 1 = one swarm |
| `obstacle_hits` | Drones touching an obstacle this step | 0 |
| `min_clearance` | Smallest gap between a drone's body and an obstacle (m) | > 0 |
| `arrived` | Drones inside the goal radius | all |
| `time_to_90pct_arrived` | Seconds until 90% of the swarm reached the goal | lower is better |
| `map_coverage_mean` / `min` / `team` | Fraction of the arena known by the average drone / the least-informed drone / the whole team | higher is better |
| `map_accuracy` | Known cells classified correctly (vs ground truth) | > 99% |
| `comm_links` / `comm_groups` | Radio links in range / connected radio groups | 1 group = everyone reachable |
| `teammates_known` / `info_age` | How many teammates each drone knows about, and how old that information is (s) | |
| `delivered` / `dropped` / `kbytes_sent` | Radio traffic | |
| `plans` / `plan_ms_mean` / `blocked_replans` | Planner activity and cost | |

Planned metrics for later phases: victims found, time to find,
messages sent, path length, formation error, energy use.

---

## Benchmark results

Phase 1, 3000 steps (150 s simulated) per run, 5 seeds per swarm size:

| Drones | Collisions | Closest approach | Local alignment | Sim speed |
|---|---|---|---|---|
| 20 | **0** | ≥ 1.57 m | 0.97–0.99 | ~680 steps/s |
| 30 | **0** | ≥ 1.51 m | 0.97–0.98 | ~630 steps/s |
| 50 | **0** | ≥ 1.55 m | 0.97–0.98 | ~500 steps/s |

~500 steps/s is about **25× real time** on an Intel i5-8365U laptop with no GPU.

### Phase 2: avoidance benchmark

50 drones, 5 seeds per row, 2400 steps (120 s simulated) per run. Drones
start on the left and must reach the goal on the right.
Reproduce with `python -m experiments.benchmark_avoidance`.

| Scenario | Method | Runs with no contact at all | Drone collisions (avg) | Obstacle hits (avg) | Min clearance (m) | Arrived | Time for 90% to arrive (s) |
|---|---|---|---|---|---|---|---|
| forest | none | 0/5 | 4.2 | 2688 | touching | 98% | 36.0 |
| forest | apf | 0/5 | 0 | 929 | touching | 99% | 51.9 |
| forest | **orca** | **5/5** | **0** | **0** | 0.30 | 98% | **44.6** |
| forest | hybrid | 5/5 | 0 | 0 | 0.32 | 98% | 61.0 |
| corridor | none | 0/5 | 0 | 4616 | touching | 99% | 51.1 |
| corridor | apf | 0/5 | 0 | 454 | touching | 98% | 53.4 |
| corridor | **orca** | **5/5** | **0** | **0** | 0.28 | 98% | **54.8** |
| corridor | hybrid | 5/5 | 0 | 0 | 0.30 | 99% | 57.6 |
| random | none | 0/5 | 0 | 5306 | touching | 96% | 41.3 |
| random | apf | 0/5 | 0 | 794 | touching | 99% | 47.8 |
| random | **orca** | **5/5** | **0** | **0** | 0.28 | 96% | **46.1** |
| random | hybrid | 5/5 | 0 | 0 | 0.30 | 97% | 53.7 |

Hits and collisions are counted in drone-steps (one touch lasting 5 steps
counts as 5).

What the numbers show:
- **none** is fastest only because hard physics lets drones scrape along
  obstacles, with thousands of contacts.
- **apf** cuts contacts by 80–90% but never reaches zero: in a crowd, neighbours
  push drones into obstacles. This is the classic weakness of potential fields.
- **orca** gives zero contacts in all 15 runs and is the fastest safe method.
  It keeps exactly the 0.3 m safety margin it is configured for.
- **hybrid** is equally safe but slower. The minimum clearance is the same
  because ORCA, not APF, decides the closest approaches.
- 96–99% arrival rather than 100%: the last 1–2 drones are usually squeezed
  out of the crowded goal circle, not stuck.

### Phase 3: communication benchmark

`survey` scenario, 50 drones, 3 seeds per row, 120 s simulated. Coverage is
the fraction of the arena a drone has classified in its **own** map.
Reproduce with `python -m experiments.benchmark_comms`.

| Setting | Average drone's map at 30 s | Time to 50% | Time to 80% | Time to 90% | Final: average / worst drone | Map accuracy | Traffic |
|---|---|---|---|---|---|---|---|
| no sharing | 17% | 119 s (2 of 3 runs) | never | never | 46% / 31% | 99.8% | 120 kB/s |
| single-hop, 0% loss | 54% | 36 s | 62 s | 69 s | 94% / 58% | 99.9% | 63 kB/s |
| **multi-hop, 0% loss** | **85%** | **3 s** | **20 s** | **49 s** | 95% / 90% | 99.8% | 186 kB/s |
| multi-hop, 20% loss | 84% | 4 s | 24 s | 51 s | 95% / 89% | 99.9% | 186 kB/s |
| multi-hop, 50% loss | 81% | 4 s | 40 s | 57 s | 95% / 86% | 99.9% | 186 kB/s |
| multi-hop, 80% loss | 72% | 16 s | 44 s | 61 s | 94% / 70% | 99.9% | 186 kB/s |

All 18 runs: 0 drone collisions, 0 obstacle hits.

What the numbers show:
- **Sharing is the difference between 17% and 85%** of the arena known per drone after 30 s.
- **Relaying matters as much as sharing**: single-hop drones only learn from
  their own flock, so the worst drone ends at 58% vs 90% with multi-hop.
- **The swarm tolerates heavy packet loss.** 50% loss costs little, because
  the same information arrives by many paths and anti-entropy repairs gaps.
  Only at 80% loss does it slow down noticeably, and every setting still ends
  above 94%.
- **Traffic** is about 3.7 kB/s per drone with multi-hop sharing (≈ 30 kbit/s),
  within reach of common low-power drone radios. "No sharing" still sends
  beacon tables, which is why it isn't zero.

### Phase 4: planning benchmark

`maze` scenario, 30 drones, 3 seeds per row, 200 s simulated.
Reproduce with `python -m experiments.benchmark_planning`.

| Setting | Arrived | Time for 90% to arrive | Drone collisions | Obstacle hits | ms per plan | Sim speed |
|---|---|---|---|---|---|---|
| no planning (straight at goal) | **0%** | never | 0 | 0 | n/a | 74 steps/s |
| plan, own map only | 100% | 148 s | 0 | 0 | 7.0 | 57 steps/s |
| **plan + shared maps** | **100%** | **107 s** | 0 | 0 | 7.1 | 44 steps/s |
| A\* + shared maps | 100% | 107 s | 0 | 0 | 34.2 | 27 steps/s |

What the numbers show:
- **Without planning the swarm never escapes the first wall.** Boids + ORCA
  head straight for the goal and get trapped in the local minimum.
- **Planning solves the maze every time**, with no collisions or hits, even
  though every drone starts knowing nothing about the maze.
- **Shared maps make the swarm 28% faster** (107 s vs 148 s): when the front
  drones discover a wall, the drones behind replan before they reach it.
- **A\* and the cost field give the same result** (same arrival time; tests
  prove equal path costs), but the cost field is about 5× cheaper per plan.

---

## Project structure

```
autonomous-drone-swarm/
├── configs/
│   └── default.yaml          all tunable parameters
├── docs/
│   └── PLAN.md               detailed roadmap and reading list
├── swarm/
│   ├── main.py               command-line entry point
│   ├── config.py             typed YAML config loader
│   ├── mathutil.py           vectorised 2D vector helpers
│   ├── metrics.py            swarm metrics
│   ├── agents/
│   │   ├── state.py          swarm state (positions, velocities, ids), spawning
│   │   └── sensors.py        lidar range sensor, nearest-obstacle detections
│   ├── mapping/
│   │   └── occupancy.py      per-drone log-odds grids, merging, ground truth
│   ├── comms/
│   │   └── network.py        radio model, beacon tables, message delivery
│   ├── planning/
│   │   ├── grid.py           cost map, A*, backward-Dijkstra cost field, path extraction
│   │   └── planner.py        per-drone replanning and string-pulling path following
│   ├── sim/
│   │   ├── world.py          arena, walls, obstacles
│   │   ├── obstacles.py      circle/box geometry: distance, raycast, collision
│   │   ├── scenarios.py      open / forest / corridor / random / survey / maze layouts
│   │   ├── dynamics.py       point-mass integration with limits
│   │   ├── neighbors.py      KD-tree neighbour queries
│   │   └── engine.py         simulation loop, recorder
│   ├── control/
│   │   ├── base.py           Controller interface, StepContext
│   │   ├── boids.py          boids controller (+ APF obstacle term)
│   │   ├── orca.py           ORCA safety layer and 2D LP solver
│   │   └── factory.py        builds none / apf / orca / hybrid stacks
│   └── viz/
│       ├── render2d.py       Pygame real-time viewer
│       └── plots.py          Matplotlib trajectory and metric plots
├── experiments/
│   ├── benchmark_avoidance.py   method x scenario x seed comparison
│   ├── benchmark_comms.py       sharing / relay / packet-loss comparison
│   └── benchmark_planning.py    planning on/off, own vs shared map, A* vs field
├── tests/
│   ├── test_core.py          Phase 0-1 tests
│   ├── test_phase2.py        Phase 2 tests
│   ├── test_phase3.py        Phase 3 tests
│   └── test_phase4.py        Phase 4 tests
├── requirements.txt
└── pytest.ini
```

Planned additions: `swarm/formation/`, `swarm/missions/` and `swarm/rl/`.

---

## Testing

```bash
python -m pytest            # 79 tests, ~1.5 min
python -m pytest -v         # verbose
```

The tests cover:
- vector maths, config loading and rejection of unknown keys
- dynamics limits and walls
- spawn spacing
- each boids rule individually
- determinism (same seed gives the same result)
- full 20- and 50-drone flights: zero collisions, the flock aligns and keeps
  flying, every drone stays in bounds
- adding and removing drones
- an offscreen viewer smoke test
- *Phase 2:* signed distance and raycasting against known answers and a
  brute-force reference, plus a regression test for the parallel-beam bug
- *Phase 2:* lidar nearest-point detection, the APF direction, and the LP solver
  (unconstrained and constrained cases)
- *Phase 2:* ORCA: two drones flying head-on never touch; a drone aimed at a
  wall stops before it
- *Phase 2:* every scenario has an obstacle-free spawn area and goal
- *Phase 2:* end to end, 30 drones cross the forest and the corridor with 0 collisions,
  0 obstacle hits and ≥ 90% arrived; the CLI headless run and viewer key handling also work
- *Phase 3:* lidar marks free space and walls correctly, never cells behind a
  wall; values stay clamped; a sweep maps a known world at > 97% accuracy
- *Phase 3:* merge rules: the more confident value wins, merging is idempotent,
  and duplicates keep the strongest value; messages send new cells first, then
  a refresh sample; single-hop sends only self-observed cells
- *Phase 3:* radio: in range → heard, out of range → never; 100% loss
  delivers nothing; latency delays delivery exactly; positions **and** map
  cells cross A → B → C only when relay is on
- *Phase 3:* end to end: sharing raises every drone's coverage by > 30 points;
  radio randomness never changes how drones fly; adding or removing drones
  resizes maps and radio; config, CLI and viewer keys all work
- *Phase 4:* cost map layers; A\* goes straight in open space, around walls,
  returns nothing when unreachable and never cuts corners
- *Phase 4:* the cost field is 0 at the goal and infinite inside a closed box;
  on 6 random maps it finds paths **exactly** as cheap as A\*
- *Phase 4:* the obstacle revision counter changes only when the obstacle
  layout changes; a wall appearing on a drone's path triggers a replan
  that avoids it; steering targets never cross a known obstacle
- *Phase 4:* end to end, in the maze 15 drones without planning all get stuck,
  while with planning ≥ 14 arrive, with 0 collisions and 0 hits; CLI and viewer work

---

## Roadmap

Full details are in [docs/PLAN.md](docs/PLAN.md).

| # | Phase | What will be built | Status |
|---|---|---|---|
| 0 | Setup | Config, engine, metrics, tests | ✅ |
| 1 | Boids | Flocking with 20–50 drones, viewer, plots | ✅ |
| 2 | Obstacles and avoidance | Obstacles, lidar, APF, ORCA safety layer, hybrid, forest / corridor / random scenarios, benchmark | ✅ |
| 3 | Communication | Radio with range / loss / latency / bandwidth, beacon tables, per-drone occupancy maps, multi-hop sharing, benchmark | ✅ |
| 4 | Path planning | Cost map, A\*, backward-Dijkstra cost field, optimistic replanning, string-pulling path following, maze, benchmark | ✅ |
| 5 | Formation flying | Leader-follower, virtual structure, line / V / circle switching, passing through gaps | 🔜 |
| 6 | Search and rescue | Area partitioning, frontier exploration, PSO victim search, CBBA task allocation, battery | 📋 |
| 7 | Terrain and threats | Heightmap cost, threat zones, PSO multi-drone route planner | 📋 |
| 8 | Reinforcement learning | PettingZoo env, shared-parameter MAPPO, curriculum learning; policies for avoidance, flocking, formation and search | 📋 |
| 9 | Experiments and demo | Scaling, communication-loss and drone-failure tests, RL vs. classical, demo video | 📋 |

---

## Algorithms

| Algorithm | Used for | Status |
|---|---|---|
| Boids (Reynolds) | Flocking | ✅ |
| Prioritised acceleration allocation | Combining behaviours safely | ✅ |
| KD-tree neighbour search | Fast local sensing | ✅ |
| Ray casting (circle + slab intersection) | Lidar simulation | ✅ |
| Artificial potential fields | Obstacle avoidance baseline | ✅ |
| ORCA / velocity obstacles + 2D linear programming | Drone–drone and drone–obstacle collision prevention | ✅ |
| Log-odds occupancy-grid mapping | Each drone's own map | ✅ |
| Gossip / epidemic dissemination with anti-entropy | Multi-hop map and position sharing | ✅ |
| A\* (octile heuristic) | Path planning, reference implementation | ✅ |
| Backward Dijkstra cost-to-go field (the quantity D\* Lite maintains) | Fast per-drone planning and re-attachment | ✅ |
| Line-of-sight string pulling | Smooth path following | ✅ |
| Incremental D\* Lite | Re-using previous searches | 📋 |
| Leader-follower, virtual structure, consensus | Formation control | 📋 Phase 5 |
| Frontier exploration, Voronoi coverage | Area search | 📋 Phase 6 |
| Particle Swarm Optimization | Victim search, multi-drone route planning | 📋 Phases 6–7 |
| CBBA / auctions | Task allocation | 📋 Phase 6 |
| MAPPO (multi-agent PPO) | Learned policies for every behaviour | 📋 Phase 8 |

---

## Hardware and design decisions

| Decision | Reason |
|---|---|
| **2D world** | All algorithms carry over to 3D later; much faster to build, train and debug |
| **Point-mass dynamics** | Enough realism for swarm algorithms without quadrotor complexity |
| **Python instead of Unity** | The development laptop (i5-8365U, 16 GB RAM, Intel UHD 620, no CUDA GPU, very little free disk) can't run Unity well; Python also gives native PyTorch for RL |
| **Vectorised NumPy** | 50 drones simulate at ~25× real time on a laptop CPU |
| **One controller interface** | Hand-written and RL controllers are interchangeable and comparable |

### RL on a CPU-only laptop (Phase 8 plan)

- Small policy networks (< 100k parameters) shared by all drones
- Training without rendering, in a vectorised environment
- Curriculum: start with 5–10 drones, then scale to 50
- One behaviour at a time, warm-started from earlier checkpoints
- Free Google Colab / Kaggle GPUs for runs that take too long

---

## Development workflow

- `main`: stable releases
- `development`: active work (all phases are built here)
- Every phase must pass the full test suite and a multi-seed headless
  benchmark before it is committed.

---

## References

- C. Reynolds, *Flocks, Herds, and Schools: A Distributed Behavioral Model*, SIGGRAPH 1987
- C. Reynolds, *Steering Behaviors for Autonomous Characters*, GDC 1999
- R. Olfati-Saber, *Flocking for Multi-Agent Dynamic Systems*, IEEE TAC 2006
- S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics* (occupancy grid mapping, ch. 9), MIT Press 2005
- A. Demers et al., *Epidemic Algorithms for Replicated Database Maintenance* (gossip, anti-entropy), PODC 1987
- O. Khatib, *Real-Time Obstacle Avoidance for Manipulators and Mobile Robots* (potential fields), IJRR 1986
- J. van den Berg et al., *Reciprocal n-Body Collision Avoidance (ORCA)*, ISRR 2011
- RVO2 Library (reference ORCA implementation), UNC Chapel Hill
- P. Hart, N. Nilsson, B. Raphael, *A Formal Basis for the Heuristic Determination of Minimum Cost Paths* (A\*), IEEE TSSC 1968
- E. Dijkstra, *A Note on Two Problems in Connexion with Graphs*, Numerische Mathematik 1959
- S. Koenig & M. Likhachev, *D\* Lite*, AAAI 2002
- W. Ren & R. Beard, *Distributed Consensus in Multi-Vehicle Cooperative Control*, 2008
- J. Kennedy & R. Eberhart, *Particle Swarm Optimization*, ICNN 1995
- B. Yamauchi, *A Frontier-Based Approach for Autonomous Exploration*, CIRA 1997
- J. Cortés et al., *Coverage Control for Mobile Sensing Networks*, IEEE TRA 2004
- H.-L. Choi et al., *Consensus-Based Decentralized Auctions (CBBA)*, IEEE TRO 2009
- C. Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)*, NeurIPS 2022

---

**Author:** Thilina Samarasinghe
