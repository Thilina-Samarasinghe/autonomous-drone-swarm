"""Typed configuration loaded from YAML, with safe defaults for every field."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SimConfig:
    dt: float = 0.05          # seconds per physics step
    seed: int = 42
    steps: int = 2000         # default length of a headless run


@dataclass
class WorldConfig:
    width: float = 200.0      # metres
    height: float = 120.0
    boundary_margin: float = 12.0


@dataclass
class DroneConfig:
    count: int = 30
    radius: float = 0.5               # physical body radius (m)
    max_speed: float = 8.0            # m/s
    cruise_speed: float = 5.0         # preferred flocking speed
    max_accel: float = 6.0            # m/s^2
    perception_radius: float = 15.0   # local sensing / neighbour range
    separation_radius: float = 4.0    # personal-space radius
    spawn_spacing: float = 3.0        # min distance between spawned drones
    spawn_speed: float = 0.5          # drones start near hover


@dataclass
class BoidsConfig:
    separation_weight: float = 2.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 0.8
    boundary_weight: float = 3.0
    goal_weight: float = 1.2


@dataclass
class SensorConfig:
    num_rays: int = 36        # lidar beams spread over 360 degrees
    range: float = 20.0       # metres


@dataclass
class AvoidanceConfig:
    method: str = "orca"              # none | apf | orca | hybrid
    apf_weight: float = 1.5           # potential-field obstacle repulsion weight
    apf_influence: float = 8.0        # obstacles closer than this repel (m)
    orca_time_horizon: float = 2.0    # look-ahead for drone-drone constraints (s)
    orca_obstacle_horizon: float = 1.5  # look-ahead for obstacle constraints (s)
    orca_neighbor_dist: float = 15.0
    orca_max_neighbors: int = 10
    safety_margin: float = 0.3        # extra clearance added to the body radius (m)


@dataclass
class ScenarioConfig:
    name: str = "open"                # open | forest | corridor | random | survey | maze
    goal_radius: float = 15.0         # drones inside this radius count as arrived
    num_obstacles: int = 25           # used by the random scenario


@dataclass
class MappingConfig:
    enabled: bool = True
    cell_size: float = 2.0            # metres per occupancy-grid cell
    update_interval: float = 0.1      # seconds between lidar -> map updates
    l_free: float = -0.4              # log-odds added when a beam passes through a cell
    l_occ: float = 0.85               # log-odds added when a beam ends in a cell
    l_max: float = 4.0                # clamp, keeps cells able to change their mind
    known_threshold: float = 1.0      # |log-odds| above this = cell is "known"


@dataclass
class CommsConfig:
    enabled: bool = True
    range: float = 30.0               # radio range (m)
    packet_loss: float = 0.0          # probability each delivery is lost (0..1)
    latency: float = 0.1              # seconds from send to receive
    broadcast_interval: float = 0.25  # seconds between broadcasts (4 Hz)
    share_maps: bool = True           # include map cells in messages
    relay: bool = True                # re-broadcast what was learned from others (multi-hop)
    max_cells_per_message: int = 300  # bandwidth cap


@dataclass
class PlanningConfig:
    enabled: bool = False
    algorithm: str = "field"          # field (backward Dijkstra, fast) | astar
    replan_interval: float = 1.0      # min seconds between replans when the map changed
    refresh_interval: float = 5.0     # full replan at least this often (unknown space shrinks)
    unknown_cost: float = 0.3         # extra cost for unknown cells (optimistic planning)
    inflation_cost: float = 8.0       # cost of cells within the drone's body radius of an obstacle
    clearance_cost: float = 2.0       # extra cost close to obstacles (keeps paths centred)
    clearance_cells: float = 2.0      # how far that extra cost reaches (cells)
    lookahead: float = 12.0           # metres along the path to look for the steering target


@dataclass
class FormationConfig:
    enabled: bool = False
    shape: str = "v"                  # line | v | circle | box | column
    spacing: float = 5.0              # metres between neighbouring slots
    speed: float = 3.5                # leader speed (slower than cruise so followers can catch up)
    gain: float = 0.8                 # slot-tracking gain (1/s)
    adaptive: bool = True             # collapse to a column for gaps, restore afterwards
    squeeze_lookahead: float = 18.0   # how far ahead the leader checks the free width (m)
    squeeze_margin: float = 2.0       # extra width required on each side (m)
    restore_delay: float = 2.0        # passage must stay wide this long before restoring (s)
    leader_timeout: float = 3.0       # leader info older than this is ignored (s)
    heading_tau: float = 1.5          # smoothing of the formation heading (s)
    slowdown_error: float = 4.0       # leader slows when mean slot error exceeds this (m)


@dataclass
class RenderConfig:
    window_width: int = 1200
    window_height: int = 760
    fps: int = 60
    trail_length: int = 40
    steps_per_frame: int = 1


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    drones: DroneConfig = field(default_factory=DroneConfig)
    boids: BoidsConfig = field(default_factory=BoidsConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    formation: FormationConfig = field(default_factory=FormationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(target: Any, overrides: dict[str, Any], path: str = "") -> None:
    valid = {f.name: f for f in fields(target)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{path}{key}'")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Config section '{path}{key}' must be a mapping")
            _merge(current, value, f"{path}{key}.")
        elif isinstance(current, bool):
            if isinstance(value, str):
                if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                    raise ValueError(f"Config key '{path}{key}' must be true/false, got '{value}'")
                value = value.lower() in ("true", "1", "yes")
            setattr(target, key, bool(value))
        else:
            setattr(target, key, type(current)(value))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Build a Config from defaults, an optional YAML file, then optional overrides."""
    cfg = Config()
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(cfg, data)
    if overrides:
        _merge(cfg, overrides)
    return cfg
