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
