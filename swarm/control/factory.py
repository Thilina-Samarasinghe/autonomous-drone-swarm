"""Builds the controller stack for an avoidance method.

  none    boids only (drones ignore obstacles; hard physics still stops them)
  apf     boids + artificial potential field from lidar
  orca    boids wrapped in the ORCA safety layer
  hybrid  boids + APF, wrapped in ORCA (APF steers smoothly, ORCA guarantees)
"""

from __future__ import annotations

from swarm.config import Config
from swarm.control.base import Controller
from swarm.control.boids import BoidsController
from swarm.control.orca import OrcaSafetyLayer
from swarm.formation.controller import FormationController

METHODS = ("none", "apf", "orca", "hybrid")


def build_controller(cfg: Config) -> Controller:
    method = cfg.avoidance.method
    if method not in METHODS:
        raise ValueError(f"Unknown avoidance method '{method}'. Choose from {METHODS}")
    use_apf = method in ("apf", "hybrid")
    if cfg.formation.enabled:
        boids = FormationController(cfg.boids, cfg.drones, cfg.avoidance, cfg.formation, use_apf)
    else:
        boids = BoidsController(cfg.boids, cfg.drones, cfg.avoidance, use_apf=use_apf)
    if method in ("orca", "hybrid"):
        return OrcaSafetyLayer(boids, cfg.drones, cfg.avoidance)
    return boids
