"""Real-time top-down Pygame viewer.

Controls
  SPACE pause / resume     .  single step (while paused)
  1-6  scenario: open/forest/corridor/random/survey/maze
  K  path planning on/off (yellow = selected drone's path)
  F  formation: off > line > v > circle > box > column
  M  cycle avoidance: none > apf > orca > hybrid
  L  links: radio > sensing > off
  G  map overlay: selected drone > team > off
  TAB  select next drone (ghosts = what it believes)
  N  map sharing on/off     P  packet loss 0/20/50/80%
  C  perception circles     T  trails    V  lidar beams
  LMB set shared goal       RMB clear goal
  + / -  add / remove 5 drones
  Q/A separation  W/S alignment  E/D cohesion  (+/- 0.1)
  [ / ]  slower / faster sim    R reset    H help    ESC quit
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pygame

from swarm.control.factory import METHODS
from swarm.formation.shapes import SHAPES
from swarm.mathutil import norms
from swarm.sim.engine import Simulation
from swarm.sim.scenarios import SCENARIOS

BG = (16, 20, 28)
ARENA = (24, 30, 42)
MARGIN = (40, 48, 64)
LINK = (60, 170, 190)
TRAIL = (90, 110, 150)
TEXT = (220, 226, 235)
GOAL = (230, 70, 70)
RADIUS = (70, 90, 120)
RADIO = (70, 200, 140)
DROP = (235, 70, 70)
SELECT = (255, 255, 255)
GHOST = (255, 255, 255)
PATH = (250, 215, 70)
SLOT = (200, 140, 255)
LEADER = (255, 90, 200)
TRAIL_L = (120, 80, 140)
OBSTACLE = (120, 128, 145)
OBSTACLE_EDGE = (170, 178, 195)
BEAM = (38, 44, 58)
BEAM_HIT = (240, 110, 90)
BAD = (250, 120, 120)
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
        pygame.display.set_caption("Autonomous Drone Swarm - Phase 5: Formation flying")
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
        self.link_mode = "radio"          # radio | sensing | off
        self.map_mode = "off"             # off | drone | team
        self.selected = 0
        self.show_circles = False
        self.show_trails = True
        self.show_help = False
        self.show_beams = False
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
            modes = ("radio", "sensing", "off")
            self.link_mode = modes[(modes.index(self.link_mode) + 1) % len(modes)]
        elif k == pygame.K_g:
            modes = ("drone", "team", "off")
            self.map_mode = modes[(modes.index(self.map_mode) + 1) % len(modes)]
        elif k == pygame.K_TAB:
            self.selected = (self.selected + 1) % max(self.sim.state.n, 1)
        elif k == pygame.K_n:
            self.sim.cfg.comms.share_maps = not self.sim.cfg.comms.share_maps
        elif k == pygame.K_p:
            levels = (0.0, 0.2, 0.5, 0.8)
            cur = self.sim.cfg.comms.packet_loss
            nxt = min(range(len(levels)), key=lambda i: abs(levels[i] - cur)) + 1
            self.sim.cfg.comms.packet_loss = levels[nxt % len(levels)]
        elif k == pygame.K_c:
            self.show_circles = not self.show_circles
        elif k == pygame.K_v:
            self.show_beams = not self.show_beams
        elif k == pygame.K_m:
            cur = self.sim.cfg.avoidance.method
            self.sim.set_avoidance(METHODS[(METHODS.index(cur) + 1) % len(METHODS)])
        elif k == pygame.K_f:
            cycle = (None,) + SHAPES
            cur = self.sim.formation.desired_shape if self.sim.formation is not None else None
            self.sim.set_formation(cycle[(cycle.index(cur) + 1) % len(cycle)])
        elif k == pygame.K_k:
            self.sim.set_planning(not self.sim.cfg.planning.enabled)
        elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
            self.sim.cfg.scenario.name = SCENARIOS[k - pygame.K_1]
            self.sim.reset()
            self.trails.clear()
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
        self._draw_map(rect)
        m = w.boundary_margin * self.scale
        pygame.draw.rect(self.screen, MARGIN, rect.inflate(-2 * m, -2 * m), 1)
        pygame.draw.rect(self.screen, MARGIN, rect, 2)
        for cx, cy, r in w.obstacles.circles:
            c = self.to_screen(np.array([cx, cy]))
            pygame.draw.circle(self.screen, OBSTACLE, c, r * self.scale)
            pygame.draw.circle(self.screen, OBSTACLE_EDGE, c, r * self.scale, 1)
        for x0, y0, x1, y1 in w.obstacles.boxes:
            tl_o = self.to_screen(np.array([x0, y1]))
            br_o = self.to_screen(np.array([x1, y0]))
            box = pygame.Rect(*tl_o, *(br_o - tl_o))
            pygame.draw.rect(self.screen, OBSTACLE, box)
            pygame.draw.rect(self.screen, OBSTACLE_EDGE, box, 1)
        if self.sim.goal is not None:
            g = self.to_screen(self.sim.goal)
            pygame.draw.circle(self.screen, (90, 50, 55), g,
                               self.sim.cfg.scenario.goal_radius * self.scale, 1)

    def _draw_map(self, rect: pygame.Rect) -> None:
        maps = self.sim.maps
        if self.map_mode == "off" or maps is None:
            return
        if self.map_mode == "drone":
            values = maps.L[min(self.selected, maps.n - 1)]
        else:
            values = maps.team_map()
        grid = maps.grid(values)[::-1]                       # row 0 = top of screen
        thr = self.sim.cfg.mapping.known_threshold
        rgba = np.zeros(grid.shape + (4,), dtype=np.uint8)
        free, occ = grid <= -thr, grid >= thr
        rgba[free] = (40, 90, 70, 150)
        rgba[occ] = (235, 150, 60, 230)
        surf = pygame.image.frombuffer(rgba.tobytes(), (maps.gw, maps.gh), "RGBA")
        self.screen.blit(pygame.transform.scale(surf, (rect.width, rect.height)), rect.topleft)

    def _draw_swarm(self) -> None:
        sim = self.sim
        d = sim.cfg.drones
        pos_s = self.to_screen(sim.state.pos)

        if self.show_trails and len(self.trails) > 1:
            pts = self.to_screen(np.stack(self.trails))  # (T, N, 2)
            for idx in range(pts.shape[1]):
                pygame.draw.lines(self.screen, TRAIL, False, pts[:, idx].tolist(), 1)

        if self.link_mode == "sensing":
            for a, b in sim.neighbors.undirected():
                pygame.draw.line(self.screen, LINK, pos_s[a], pos_s[b], 1)
        elif self.link_mode == "radio" and sim.comms is not None:
            for a, b in sim.comms.links:
                pygame.draw.line(self.screen, RADIO, pos_s[a], pos_s[b], 1)
            for a, b in sim.comms.recent_drops:
                if a < len(pos_s) and b < len(pos_s):
                    pygame.draw.line(self.screen, DROP, pos_s[a], pos_s[b], 2)

        if self.show_beams:
            scan = sim.scan
            ends = sim.state.pos[:, None, :] + scan.dirs[None] * scan.dist[:, :, None]
            ends_s = self.to_screen(ends)
            hits = scan.hits
            for k in range(len(pos_s)):
                for r in range(ends_s.shape[1]):
                    col = BEAM_HIT if hits[k, r] else BEAM
                    pygame.draw.line(self.screen, col, pos_s[k], ends_s[k, r], 1)

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

        # formation: leader trail, follower slots, leader marker
        fm = sim.formation
        if fm is not None and sim.state.n:
            if len(fm.trail) > 1:
                pygame.draw.lines(self.screen, TRAIL_L, False,
                                  self.to_screen(np.array(fm.trail[-300:])).tolist(), 1)
            ft = getattr(sim, "last_formation_targets", None)
            if ft is not None and len(ft.slot) == sim.state.n:
                for k in range(sim.state.n):
                    if k == ft.leader or np.isnan(ft.slot[k, 0]):
                        continue
                    sp = self.to_screen(ft.slot[k])
                    col = PATH if fm.following_trail[k] else SLOT
                    pygame.draw.rect(self.screen, col, (sp[0] - 3, sp[1] - 3, 6, 6), 1)
            pygame.draw.circle(self.screen, LEADER, pos_s[fm.leader], size + 4, 2)

        # selected drone: planned path + steering target
        if sim.planner is not None and sim.state.n:
            k = min(self.selected, sim.state.n - 1)
            path = sim.planner.paths[k] if k < sim.planner.n else None
            if path is not None and len(path) > 1:
                pygame.draw.lines(self.screen, PATH, False, self.to_screen(path).tolist(), 2)
            if sim.planner.carrots is not None and k < len(sim.planner.carrots):
                c = self.to_screen(sim.planner.carrots[k])
                pygame.draw.line(self.screen, PATH, pos_s[k], c, 1)
                pygame.draw.circle(self.screen, PATH, c, 4)

        # selected drone: radio range + where it BELIEVES the others are
        if sim.state.n:
            k = min(self.selected, sim.state.n - 1)
            pygame.draw.circle(self.screen, SELECT, pos_s[k], size + 5, 2)
            if sim.comms is not None:
                pygame.draw.circle(self.screen, (60, 110, 90), pos_s[k],
                                   sim.cfg.comms.range * self.scale, 1)
                belief = sim.comms.table[k]
                seen = ~np.isnan(belief[:, 2])
                seen[k] = False
                ghosts = self.to_screen(belief[seen, :2])
                ages = sim.time - belief[seen, 2]
                for gp, age in zip(ghosts, ages):
                    fade = int(np.clip(255 - age * 12, 60, 255))
                    pygame.draw.circle(self.screen, (fade, fade, fade), gp, 3, 1)

    def _draw_hud(self) -> None:
        sim, m, w = self.sim, self.sim.metrics, self.sim.cfg.boids
        x0 = self.screen.get_width() - HUD_W
        pygame.draw.rect(self.screen, (12, 15, 22), (x0, 0, HUD_W, self.screen.get_height()))
        lines = [
            ("PHASE 5 - FORMATIONS", None),
            ("", None),
            (f"scenario  {sim.cfg.scenario.name:>11s}", (150, 200, 255)),
            (f"avoidance {sim.cfg.avoidance.method:>11s}", (150, 200, 255)),
            (f"planning  {(sim.cfg.planning.algorithm if sim.planner else 'off'):>11s}", (150, 200, 255)),
            (f"formation {(sim.formation.active_shape if sim.formation else 'off'):>11s}", (150, 200, 255)),
            ("", None),
            (f"time {m.time:7.1f} s   drones {m.n:3d}", None),
            (f"fps  {self.clock.get_fps():7.1f}     speed {self.steps_per_frame:2d}x", None),
            ("", None),
            (f"min sep      {m.min_separation:8.2f} m", BAD if m.collisions else None),
            (f"collisions now/total {m.collisions:3d}/{sim.total_collisions:<4d}", BAD if m.collisions else None),
            (f"obst. hits now/total {m.obstacle_hits:3d}/{sim.total_obstacle_hits:<4d}", BAD if m.obstacle_hits else None),
            (f"clearance    {m.min_clearance:8.2f} m", BAD if m.min_clearance < 0 else None),
            (f"arrived      {m.arrived:5d}/{m.n:<3d}", None),
            (f"speed {m.mean_speed:4.1f} m/s  groups {m.groups:3d}", None),
            (f"weights s/a/c {w.separation_weight:.1f}/{w.alignment_weight:.1f}/{w.cohesion_weight:.1f}", None),
            ("", None),
        ]
        im = sim.info_metrics() if (sim.maps is not None or sim.comms is not None) else None
        if im is not None:
            cfgc = sim.cfg.comms
            drop_pct = 100.0 * im.dropped / max(im.dropped + im.delivered, 1)
            k = min(self.selected, max(sim.state.n - 1, 0))
            sel_cov = (float(sim.truth.coverage(sim.maps.L[k], sim.cfg.mapping.known_threshold))
                       if sim.maps is not None else float("nan"))
            lines += [
                ("COMMS & MAP", (150, 200, 255)),
                (f"sharing  {'on' if cfgc.share_maps else 'OFF':>4s} loss {cfgc.packet_loss:4.0%}", None),
                (f"radio links {im.comm_links:5d} groups {im.comm_groups:3d}", None),
                (f"dropped      {drop_pct:7.1f}%", BAD if drop_pct > 0 else None),
                (f"traffic      {im.kbytes_sent / max(sim.time, 1e-9):6.1f} kB/s", None),
                (f"map #{k:<3d}{sel_cov:4.0%} avg {im.map_coverage_mean:4.0%} team {im.map_coverage_team:4.0%}", None),
                (f"map accuracy {im.map_accuracy:8.1%}", None),
                (f"info age     {im.info_age:7.1f} s", None),
            ]
        if sim.formation is not None:
            fs = sim.formation.stats
            fw = sim.formation.free_width
            lines += [
                (f"slot error   {fs.last_error:7.2f} m", BAD if fs.last_error > 5 else None),
                (f"in shape     {fs.in_shape_fraction:8.0%}", None),
                (f"passage {('open' if fw == float('inf') else f'{fw:4.0f} m'):>6s} sq {fs.squeezes:3d}", None),
            ]
        if sim.planner is not None:
            ps = sim.planner.stats
            lines += [
                (f"plans {ps.plans:6d}  {ps.mean_ms:5.1f} ms", None),
                (f"blocked replans {ps.blocked_replans:6d}", None),
            ]
        lines += [
            ("", None),
            ("PAUSED" if self.paused else "running", (250, 210, 60) if self.paused else (120, 220, 140)),
            ("H = help", (140, 150, 170)),
        ]
        y = 12
        if self.show_help:   # help replaces the stats so it always fits
            self.screen.blit(self.font.render("CONTROLS  (H to close)", True, TEXT), (x0 + 14, y))
            y += 28
            for line in (__doc__ or "").strip().splitlines()[3:]:
                self.screen.blit(self.small.render(line.strip(), True, (170, 180, 200)), (x0 + 10, y))
                y += 18
            return
        for text, color in lines:
            self.screen.blit(self.font.render(text, True, color or TEXT), (x0 + 14, y))
            y += 19

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
