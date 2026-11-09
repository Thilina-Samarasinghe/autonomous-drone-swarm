"""Real-time top-down Pygame viewer.

Controls
  SPACE pause / resume     .  single step (while paused)
  L  neighbour links        C  perception circles     T  trails
  LMB set shared goal       RMB clear goal
  + / -  add / remove 5 drones
  Q/A separation  W/S alignment  E/D cohesion  (weight +/- 0.1)
  [ / ]  slower / faster sim    R reset    H help    ESC quit
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pygame

from swarm.mathutil import norms
from swarm.sim.engine import Simulation

BG = (16, 20, 28)
ARENA = (24, 30, 42)
MARGIN = (40, 48, 64)
LINK = (60, 170, 190)
TRAIL = (90, 110, 150)
TEXT = (220, 226, 235)
GOAL = (230, 70, 70)
RADIUS = (70, 90, 120)
HUD_W = 260


def _speed_color(s: float, vmax: float) -> tuple[int, int, int]:
    t = float(np.clip(s / vmax, 0, 1))
    # blue (slow) -> cyan -> yellow (fast)
    stops = [(70, 120, 255), (60, 220, 220), (250, 210, 60)]
    k = min(int(t * 2), 1)
    u = t * 2 - k
    a, b = stops[k], stops[k + 1]
    return tuple(int(a[c] + (b[c] - a[c]) * u) for c in range(3))


class Viewer:
    def __init__(self, sim: Simulation):
        self.sim = sim
        rc = sim.cfg.render
        pygame.init()
        pygame.display.set_caption("Autonomous Drone Swarm - Phase 1: Boids")
        self.screen = pygame.display.set_mode((rc.window_width, rc.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 15)
        self.small = pygame.font.SysFont("consolas", 13)

        view_w = rc.window_width - HUD_W
        self.scale = min(view_w / sim.world.width, rc.window_height / sim.world.height) * 0.96
        self.offset = np.array([
            (view_w - sim.world.width * self.scale) / 2,
            (rc.window_height - sim.world.height * self.scale) / 2,
        ])

        self.paused = False
        self.show_links = True
        self.show_circles = False
        self.show_trails = True
        self.show_help = False
        self.steps_per_frame = max(1, rc.steps_per_frame)
        self.trails: deque[np.ndarray] = deque(maxlen=rc.trail_length)

    # -------------------------------------------------------------- mapping
    def to_screen(self, p: np.ndarray) -> np.ndarray:
        # world y-up -> screen y-down
        x = self.offset[0] + p[..., 0] * self.scale
        y = self.offset[1] + (self.sim.world.height - p[..., 1]) * self.scale
        return np.stack([x, y], axis=-1)

    def to_world(self, sx: float, sy: float) -> np.ndarray:
        x = (sx - self.offset[0]) / self.scale
        y = self.sim.world.height - (sy - self.offset[1]) / self.scale
        return np.clip(np.array([x, y]), 0, self.sim.world.size)

    # --------------------------------------------------------------- events
    def _handle(self, ev) -> bool:
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1 and ev.pos[0] < self.screen.get_width() - HUD_W:
                self.sim.goal = self.to_world(*ev.pos)
            elif ev.button == 3:
                self.sim.goal = None
        if ev.type != pygame.KEYDOWN:
            return True
        k, w = ev.key, self.sim.cfg.boids
        if k == pygame.K_ESCAPE:
            return False
        elif k == pygame.K_SPACE:
            self.paused = not self.paused
        elif k == pygame.K_PERIOD and self.paused:
            self._advance(1)
        elif k == pygame.K_l:
            self.show_links = not self.show_links
        elif k == pygame.K_c:
            self.show_circles = not self.show_circles
        elif k == pygame.K_t:
            self.show_trails = not self.show_trails
            self.trails.clear()
        elif k == pygame.K_h:
            self.show_help = not self.show_help
        elif k == pygame.K_r:
            self.sim.reset()
            self.trails.clear()
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.sim.add_drones(5)
            self.trails.clear()
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.sim.remove_drones(5)
            self.trails.clear()
        elif k == pygame.K_LEFTBRACKET:
            self.steps_per_frame = max(1, self.steps_per_frame - 1)
        elif k == pygame.K_RIGHTBRACKET:
            self.steps_per_frame = min(20, self.steps_per_frame + 1)
        else:
            tweaks = {
                pygame.K_q: ("separation_weight", 0.1), pygame.K_a: ("separation_weight", -0.1),
                pygame.K_w: ("alignment_weight", 0.1), pygame.K_s: ("alignment_weight", -0.1),
                pygame.K_e: ("cohesion_weight", 0.1), pygame.K_d: ("cohesion_weight", -0.1),
            }
            if k in tweaks:
                name, delta = tweaks[k]
                setattr(w, name, round(max(0.0, getattr(w, name) + delta), 2))
        return True

    def _advance(self, steps: int) -> None:
        for _ in range(steps):
            self.sim.step()
        if self.show_trails:
            if self.trails and self.trails[-1].shape != self.sim.state.pos.shape:
                self.trails.clear()
            self.trails.append(self.sim.state.pos.copy())

    # --------------------------------------------------------------- drawing
    def _draw_world(self) -> None:
        w = self.sim.world
        tl = self.to_screen(np.array([0.0, w.height]))
        br = self.to_screen(np.array([w.width, 0.0]))
        rect = pygame.Rect(*tl, *(br - tl))
        pygame.draw.rect(self.screen, ARENA, rect)
        m = w.boundary_margin * self.scale
        pygame.draw.rect(self.screen, MARGIN, rect.inflate(-2 * m, -2 * m), 1)
        pygame.draw.rect(self.screen, MARGIN, rect, 2)

    def _draw_swarm(self) -> None:
        sim = self.sim
        d = sim.cfg.drones
        pos_s = self.to_screen(sim.state.pos)

        if self.show_trails and len(self.trails) > 1:
            pts = self.to_screen(np.stack(self.trails))  # (T, N, 2)
            for idx in range(pts.shape[1]):
                pygame.draw.lines(self.screen, TRAIL, False, pts[:, idx].tolist(), 1)

        if self.show_links:
            for a, b in sim.neighbors.undirected():
                pygame.draw.line(self.screen, LINK, pos_s[a], pos_s[b], 1)

        if self.show_circles:
            r = int(d.perception_radius * self.scale)
            for p in pos_s:
                pygame.draw.circle(self.screen, RADIUS, p, r, 1)

        speeds = norms(sim.state.vel)
        size = max(4.0, d.radius * self.scale * 2.2)
        for p, v, s in zip(pos_s, sim.state.vel, speeds):
            ang = np.arctan2(-v[1], v[0])  # screen y flipped
            fwd = np.array([np.cos(ang), np.sin(ang)])
            side = np.array([-fwd[1], fwd[0]])
            tri = [p + fwd * size, p - fwd * size * 0.6 + side * size * 0.55,
                   p - fwd * size * 0.6 - side * size * 0.55]
            pygame.draw.polygon(self.screen, _speed_color(s, d.max_speed), tri)

        if sim.goal is not None:
            g = self.to_screen(sim.goal)
            pygame.draw.circle(self.screen, GOAL, g, 8, 2)
            pygame.draw.circle(self.screen, GOAL, g, 3)

    def _draw_hud(self) -> None:
        sim, m, w = self.sim, self.sim.metrics, self.sim.cfg.boids
        x0 = self.screen.get_width() - HUD_W
        pygame.draw.rect(self.screen, (12, 15, 22), (x0, 0, HUD_W, self.screen.get_height()))
        lines = [
            ("PHASE 1 - BOIDS", None),
            ("", None),
            (f"time         {m.time:8.1f} s", None),
            (f"drones       {m.n:8d}", None),
            (f"fps          {self.clock.get_fps():8.1f}", None),
            (f"sim speed    {self.steps_per_frame:7d}x", None),
            ("", None),
            (f"min sep      {m.min_separation:8.2f} m", (250, 120, 120) if m.collisions else None),
            (f"collisions   {m.collisions:8d}", (250, 120, 120) if m.collisions else None),
            (f"total coll.  {sim.total_collisions:8d}", None),
            (f"mean speed   {m.mean_speed:8.2f} m/s", None),
            (f"polarization {m.polarization:8.2f}", None),
            (f"local polar. {m.local_polarization:8.2f}", None),
            (f"groups       {m.groups:8d}", None),
            ("", None),
            (f"sep  w (Q/A) {w.separation_weight:8.2f}", None),
            (f"align w(W/S) {w.alignment_weight:8.2f}", None),
            (f"coh  w (E/D) {w.cohesion_weight:8.2f}", None),
            ("", None),
            ("PAUSED" if self.paused else "running", (250, 210, 60) if self.paused else (120, 220, 140)),
            ("H = help", (140, 150, 170)),
        ]
        y = 16
        for text, color in lines:
            self.screen.blit(self.font.render(text, True, color or TEXT), (x0 + 14, y))
            y += 20
        if self.show_help:
            y += 10
            for line in (__doc__ or "").strip().splitlines()[2:]:
                self.screen.blit(self.small.render(line.strip(), True, (160, 170, 190)), (x0 + 10, y))
                y += 17

    # ------------------------------------------------------------------ main
    def run(self, max_frames: int | None = None) -> None:
        running, frames = True, 0
        while running:
            for ev in pygame.event.get():
                running = self._handle(ev) and running
            if not self.paused:
                self._advance(self.steps_per_frame)
            self.screen.fill(BG)
            self._draw_world()
            self._draw_swarm()
            self._draw_hud()
            pygame.display.flip()
            self.clock.tick(self.sim.cfg.render.fps)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        pygame.quit()
