"""Build the research guide PDF (docs/Autonomous_Drone_Swarm_Research_Guide.pdf).

  python -m experiments.make_research_guide

What it covers: the research areas, topics and sub-topics behind the
project, the core theories (with the equations as implemented), the
technologies used, the key results, lessons learned, and how the
simulation maps onto real-world drone systems and applications, with
references. Figures come from docs/figures/ (python -m experiments.make_figures).
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

OUT = Path("docs/Autonomous_Drone_Swarm_Research_Guide.pdf")
FIG = Path("docs/figures")
AUTHOR = "Thilina Samarasinghe"
REPO = "github.com/Thilina-Samarasinghe/autonomous-drone-swarm"

# DejaVu ships with matplotlib and covers Greek letters, arrows and maths symbols
_TTF = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_TTF / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(_TTF / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(_TTF / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(_TTF / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(_TTF / "DejaVuSansMono.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402

registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY = colors.HexColor("#1a365d")
BLUE = colors.HexColor("#2b6cb0")
LIGHT = colors.HexColor("#ebf4ff")
GREY = colors.HexColor("#4a5568")
RULE = colors.HexColor("#cbd5e0")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=32, textColor=NAVY, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=13, leading=18, textColor=GREY, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=17, leading=22, textColor=NAVY, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="DV-B", fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=6, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.5, leading=13, leftIndent=14, bulletIndent=4,
                             spaceAfter=2),
    "eq": ParagraphStyle("eq", fontName="DV", fontSize=10, leading=15, leftIndent=24, textColor=NAVY,
                         backColor=LIGHT, borderPadding=(4, 6, 4, 6), spaceBefore=3, spaceAfter=7),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.3, leading=10.8),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.3, leading=10.8, textColor=colors.white),
    "caption": ParagraphStyle("caption", fontName="DV-I", fontSize=8.3, leading=11, textColor=GREY,
                              alignment=TA_CENTER, spaceAfter=10),
    "ref": ParagraphStyle("ref", fontName="DV", fontSize=8.4, leading=11.2, leftIndent=22, firstLineIndent=-22,
                          spaceAfter=3),
    "small": ParagraphStyle("small", fontName="DV", fontSize=8.5, leading=11.5, textColor=GREY),
    "toc0": ParagraphStyle("toc0", fontName="DV-B", fontSize=10.5, leading=16, leftIndent=0),
    "toc1": ParagraphStyle("toc1", fontName="DV", fontSize=9.5, leading=13, leftIndent=16),
}


# ----------------------------------------------------------------- helpers
def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def eq(text):
    return Paragraph(text, S["eq"])


def table(rows, widths, header=True, zebra=True):
    data = []
    for i, r in enumerate(rows):
        st = "cellb" if (header and i == 0) else "cell"
        data.append([Paragraph(str(c), S[st]) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7fafc")))
    t.setStyle(TableStyle(style))
    return t


def figure(path, caption, width=16.5 * cm):
    p = Path(path)
    if not p.exists():
        return [P(f"<i>(figure missing: {path} - run python -m experiments.make_figures)</i>", "small")]
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    img = Image(str(p), width=width, height=width * h / w)
    return [KeepTogether([img, Spacer(1, 3), P(caption, "caption")])]


class H(Paragraph):
    """Heading that registers itself in the table of contents."""

    def __init__(self, text, level):
        super().__init__(text, S["h1" if level == 0 else "h2"])
        self.toc_level = level


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                         bottomMargin=2 * cm, title="Autonomous Drone Swarm - Research Guide", author=AUTHOR,
                         subject="Research areas, theories, technologies and sim-to-real roadmap")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=lambda c, d: None),
                               PageTemplate("body", [frame], onPage=self._decorate)])

    def _decorate(self, canv, doc):
        canv.saveState()
        canv.setFont("DV", 7.5)
        canv.setFillColor(GREY)
        canv.drawString(2 * cm, A4[1] - 1.3 * cm, "Autonomous Drone Swarm Simulation - Research Guide")
        canv.drawRightString(A4[0] - 2 * cm, A4[1] - 1.3 * cm, AUTHOR)
        canv.setStrokeColor(RULE)
        canv.line(2 * cm, A4[1] - 1.45 * cm, A4[0] - 2 * cm, A4[1] - 1.45 * cm)
        canv.drawCentredString(A4[0] / 2, 1.2 * cm, f"Page {doc.page}")
        canv.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, H):
            text = flowable.getPlainText()
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=flowable.toc_level, closed=flowable.toc_level > 0)
            self.notify("TOCEntry", (flowable.toc_level, text, self.page, key))


W = 17 * cm   # usable width


# ------------------------------------------------------------------ content
def cover():
    today = dt.date.today().strftime("%B %Y")
    return [
        Spacer(1, 4.5 * cm),
        P("Autonomous Drone Swarm Simulation", "title"),
        Spacer(1, 10),
        P("Research Guide: Areas, Theories and Technologies,<br/>and the Path from Simulation to Real-World Systems",
          "subtitle"),
        Spacer(1, 1.6 * cm),
        table([["What this document is"],
               ["A study companion to the project: it lists the research areas, topics and sub-topics "
                "behind it, explains the key theories with the equations exactly as implemented, "
                "summarises the technologies and results, records the lessons learned, and lays out how "
                "the work transfers from simulation to real drones and real-world problems, with references."]],
              [12 * cm]),
        Spacer(1, 2.2 * cm),
        P(f"<b>{AUTHOR}</b>", "subtitle"),
        P(today, "subtitle"),
        P(REPO, "subtitle"),
        NextPageTemplate("body"),
        PageBreak(),
    ]


def toc_page():
    toc = TableOfContents()
    toc.levelStyles = [S["toc0"], S["toc1"]]
    return [P("Contents", "h1"), toc, PageBreak()]


def s1_overview():
    out = [H("1. Project overview", 0)]
    out.append(P(
        "The project is a 2D simulator of a swarm of 20–50 autonomous drones written in Python. "
        "<b>No central controller exists.</b> Each drone decides for itself from what it can sense "
        "(neighbours within 15 m, a 36-beam lidar), what it has mapped itself, and what teammates tell it "
        "over a lossy, range-limited radio. Cooperative behaviour (flocking, formations, shared maps, "
        "search and rescue) <i>emerges</i> from these local decisions. That is the defining property of "
        "swarm intelligence and the main research theme of the project."))
    out.append(P(
        "It was built in ten phases, each adding one capability and each verified by automated tests "
        "(189 in total) and by benchmarks repeated over several random seeds. The results below are "
        "measured, not assumed."))
    rows = [["Phase", "Capability", "Headline result"],
            ["0–1", "Simulation core, boids flocking", "50 drones flock with 0 collisions; local alignment 0.97"],
            ["2", "Lidar, potential fields, ORCA", "ORCA: 0 collisions and 0 obstacle contacts in all 15 runs"],
            ["3", "Radio, occupancy maps, multi-hop sharing", "Each drone knows 85% of the arena after 30 s vs 17% alone"],
            ["4", "Path planning (A*, cost field)", "No planning: 0% escape the maze; planning: 100%; shared maps 29% faster"],
            ["5", "Formation flying", "V / line / box / circle; through gaps and a maze with nobody left behind"],
            ["6", "Search and rescue", "8/8 victims found in every run (random search: 3.7/8)"],
            ["7", "Terrain and threats, PSO routes", "0 s in lethal zones; 4–5× fewer close encounters than grid routes"],
            ["8", "Multi-agent reinforcement learning", "RL + ORCA ~25% faster; RL alone not as safe as hand-written control"],
            ["9", "Fault injection and robustness", "0 collisions in 68 stress runs; 8/8 victims with 90% packet loss"]]
    out.append(table(rows, [1.3 * cm, 5.5 * cm, 10.2 * cm]))
    out.append(Spacer(1, 8))
    out.append(P("<b>Design decisions.</b> 2D world and point-mass dynamics (every algorithm transfers to 3D); "
                 "pure Python so it runs on a CPU-only laptop (Intel i5-8365U, no GPU); one controller interface "
                 "so hand-written and learned controllers are interchangeable and compared fairly; a seeded, "
                 "fixed-time-step engine so every experiment is reproducible."))
    return out


def s2_research_map():
    out = [PageBreak(), H("2. Research map: areas, topics and sub-topics", 0)]
    out.append(P("The project sits where several research fields meet. The table lists each area, the topics "
                 "studied, and the concrete sub-topics that were implemented (I) or studied for comparison (S)."))
    rows = [["Research area", "Topics", "Sub-topics (I = implemented, S = studied)"],
            ["A. Multi-agent systems and swarm intelligence",
             "Emergence, decentralisation, local interaction rules, collective behaviour",
             "Boids (I); order parameter / polarisation (I); consensus and flocking theory (S); stigmergy (S); "
             "scalability with swarm size (I)"],
            ["B. Motion control and collision avoidance",
             "Kinematic agents, steering behaviours, reactive avoidance, provable safety",
             "Point-mass dynamics with limits (I); prioritised acceleration (I); artificial potential fields (I); "
             "velocity obstacles, RVO, ORCA (I); 2D linear programming (I); control barrier functions (S)"],
            ["C. Perception and mapping",
             "Range sensing, probabilistic mapping, sensor uncertainty",
             "Lidar ray casting (I); occupancy grids with log-odds (I); ground-truth evaluation (I); "
             "sensor noise and dropout (I); SLAM (S)"],
            ["D. Networked systems and communication",
             "Wireless ad-hoc networks, information spreading, consistency",
             "Range-limited broadcast, loss, latency, bandwidth (I); beacon tables (I); gossip / epidemic "
             "protocols (I); anti-entropy (I); multi-hop relay (I); network connectivity (I)"],
            ["E. Path planning and graph search",
             "Shortest paths, heuristics, replanning in unknown space",
             "Dijkstra (I); A* with octile heuristic (I); backward cost-to-go field (I); optimistic planning (I); "
             "cost maps and inflation (I); string pulling (I); D* Lite, Theta* (S)"],
            ["F. Formation control",
             "Leader-follower, virtual structure, assignment",
             "Slot tracking with feed-forward (I); Hungarian assignment (I); breadcrumb trail following (I); "
             "passage-width estimation (I); leader re-election (I); consensus-based formation (S)"],
            ["G. Optimisation and metaheuristics",
             "Population-based search, constrained optimisation",
             "Particle Swarm Optimisation (I); inertia-weight schedule (I); hybrid PSO seeded by graph search (I); "
             "penalty methods (I); prioritised multi-agent planning (I); Douglas–Peucker (I)"],
            ["H. Cooperative missions and task allocation",
             "Exploration, coverage, search theory, task assignment, energy",
             "Frontier exploration (I); dynamic Voronoi partitioning (I); consensus auctions / CBBA (I, simplified); "
             "battery-aware return (I); signal source seeking (I)"],
            ["I. Machine learning and multi-agent RL",
             "MDPs, policy gradients, centralised training with decentralised execution",
             "PPO (I); GAE (I); MAPPO-style centralised critic (I); parameter sharing (I); reward shaping (I); "
             "curriculum learning (I); generalisation across swarm sizes (I); sim-to-real transfer (S)"],
            ["J. Reliability and fault tolerance",
             "Graceful degradation, failure handling",
             "Fault injection (I); leader election (I); robustness to radio loss and sensor noise (I); "
             "stuck detection and recovery (I)"],
            ["K. Software engineering and experimental method",
             "Reproducible research, testing, performance",
             "Unit and end-to-end tests (I); multi-seed benchmarks (I); profiling and vectorisation (I); "
             "configuration management (I); visualisation (I)"]]
    out.append(table(rows, [3.6 * cm, 4.4 * cm, 9.0 * cm]))
    return out


def s3_theories():
    out = [PageBreak(), H("3. Core theories and how they were applied", 0)]
    out.append(P("For each theory: the idea, the equation as it is implemented in the code, where it lives, and "
                 "what the project taught about it. Vectors are 2D; <i>dt</i> = 0.05 s."))

    def theory(title, idea, equations, where, learned):
        block = [KeepTogether([P(title, "h2"), P(idea)])]
        block += [eq(e) for e in equations]
        block.append(P(f"<b>In the project:</b> {where}"))
        block.append(P(f"<b>What we learned:</b> {learned}"))
        return block

    out += theory(
        "3.1 Point-mass dynamics and steering behaviours",
        "Each drone is a point with position <b>p</b> and velocity <b>v</b>, limited to |v| ≤ 8 m/s and "
        "|a| ≤ 6 m/s². A controller only chooses an acceleration; semi-implicit Euler integration advances "
        "the state. Reynolds' steering idea: steering = desired velocity − current velocity.",
        ["<b>v</b> ← clamp(<b>v</b> + clamp(<b>a</b>, a<sub>max</sub>)·dt, v<sub>max</sub>);  "
         "<b>p</b> ← <b>p</b> + <b>v</b>·dt",
         "<b>a</b><sub>steer</sub> = clamp(v<sub>des</sub>·<b>û</b> − <b>v</b>, a<sub>max</sub>)"],
        "<font name='DVM'>swarm/sim/dynamics.py</font>, <font name='DVM'>swarm/control/boids.py</font>",
        "Limits matter: a drone that is told to go somewhere it physically cannot brake for will collide, "
        "whatever the algorithm says (Phase 1: drones spawned at full speed collided before they could stop).")
    out += theory(
        "3.2 Boids flocking and prioritised acceleration",
        "Three local rules produce a flock: separation (avoid crowding), alignment (match neighbours' heading) "
        "and cohesion (move toward the local centre). Reynolds' prioritised acceleration allocation gives "
        "safety-critical rules first claim on the limited acceleration budget.",
        ["separation: Σ<sub>j</sub> (p<sub>i</sub> − p<sub>j</sub>) / d<sub>ij</sub>²  over neighbours with "
         "d<sub>ij</sub> &lt; 4 m (anticipated 0.6 s ahead)",
         "alignment: mean neighbour velocity;  cohesion: (centroid − p<sub>i</sub>) / r<sub>perception</sub> · a<sub>max</sub>",
         "polarisation (order parameter) φ = |(1/N) Σ v<sub>i</sub>/|v<sub>i</sub>| |  ∈ [0, 1]"],
        "<font name='DVM'>swarm/control/boids.py</font>, metric in <font name='DVM'>swarm/metrics.py</font>",
        "A naive weighted sum let weak rules cancel collision avoidance. Prioritised allocation plus "
        "anticipatory separation (reacting to where a neighbour <i>will</i> be) removed every collision. "
        "A cohesion rule written as “desired − current” acted as a brake and halved the flock's speed.")
    out += theory(
        "3.3 Artificial potential fields (APF)",
        "Obstacles create a repulsive force that grows as the drone gets closer (Khatib, 1986). Cheap and "
        "smooth, but known to suffer from local minima and to give no safety guarantee.",
        ["F<sub>rep</sub> = a<sub>max</sub> · ((d<sub>0</sub> − d) / (d<sub>0</sub> − r))² · <b>n</b>  "
         "for d &lt; d<sub>0</sub> = 8 m  (plus a tangential term to slide around obstacles)"],
        "obstacle term in <font name='DVM'>swarm/control/boids.py</font>",
        "APF alone never reached zero obstacle contact (in a crowd it must trade drone separation against "
        "obstacle clearance) and, without planning, it traps the swarm at the first wall of a maze.")
    out += theory(
        "3.4 Velocity obstacles and ORCA",
        "A velocity obstacle is the set of velocities that lead to a collision within a time horizon τ. ORCA "
        "(Optimal Reciprocal Collision Avoidance) turns each neighbour into one half-plane constraint on the "
        "drone's own velocity, with each drone taking <i>half</i> of the avoidance effort, so no communication "
        "is needed. A small 2D linear program picks the allowed velocity closest to the preferred one.",
        ["ORCA<sub>A|B</sub> = { v : (v − (v<sub>A</sub> + ½<b>u</b>)) · <b>n</b> ≥ 0 }",
         "v<sub>new</sub> = argmin<sub>v ∈ ∩ ORCA</sub> |v − v<sub>pref</sub>|   subject to |v| ≤ v<sub>max</sub>",
         "obstacle point at distance d:  <b>û</b>·v ≤ (d − r) / τ<sub>obs</sub>   (full effort; obstacles do not move)"],
        "<font name='DVM'>swarm/control/orca.py</font> (a port of the RVO2 linear-programming solver)",
        "ORCA gave 0 collisions and 0 obstacle contacts in every run, and was the fastest safe method. Wrapped "
        "around any other controller (boids, formations, even a learned policy) it acts as a safety layer.")
    out += theory(
        "3.5 Occupancy-grid mapping (Bayesian, log-odds)",
        "The world is a grid; each cell stores the probability that it is occupied, kept as log-odds so a "
        "Bayesian update becomes an addition. Beams that pass through a cell are evidence of free space; a beam "
        "ending in a cell is evidence of an obstacle.",
        ["l(m | z<sub>1:t</sub>) = l(m | z<sub>1:t−1</sub>) + log [ p(m | z<sub>t</sub>) / (1 − p(m | z<sub>t</sub>)) ]",
         "implemented: l ← clamp(l + l<sub>occ</sub> or l<sub>free</sub>, ±4),  l<sub>occ</sub> = 0.85, l<sub>free</sub> = −0.4;  "
         "known if |l| ≥ 1"],
        "<font name='DVM'>swarm/mapping/occupancy.py</font>",
        "Maps were 99.8%+ accurate against ground truth. Clamping keeps cells able to change their mind; merging "
        "maps by “more confident wins” is idempotent, so information relayed many times is never double-counted.")
    out += theory(
        "3.6 Gossip protocols and anti-entropy",
        "Epidemic (gossip) algorithms spread information by repeated exchanges with neighbours; anti-entropy "
        "periodically re-sends known state so that lost messages are eventually repaired (Demers et al., 1987).",
        ["merge rules: confirmations → logical OR;  best readings → maximum;  beacon / bid tables → newest timestamp",
         "each message = newly learned cells + a random sample of known cells (anti-entropy)"],
        "<font name='DVM'>swarm/comms/network.py</font>, gossip in <font name='DVM'>swarm/mission/manager.py</font>",
        "Without anti-entropy a cell whose messages were all lost was never shared again. With it, the swarm "
        "kept 81% map coverage per drone even at 50% packet loss, and found all victims at 90% loss.")
    out += theory(
        "3.7 Graph search: Dijkstra, A* and cost-to-go fields",
        "A* expands nodes in order of f = g + h, where g is the cost so far and h an admissible (never "
        "overestimating) heuristic, which guarantees an optimal path. Running Dijkstra <i>backwards</i> from "
        "the goal gives the cost-to-go from every cell, the quantity D* Lite maintains incrementally.",
        ["f(n) = g(n) + h(n);   octile heuristic h = (Δx + Δy) + (√2 − 2)·min(Δx, Δy)",
         "edge cost = length × ½ (c<sub>a</sub> + c<sub>b</sub>);  c = 1 free, +0.3 unknown, +8 inflated, ∞ obstacle"],
        "<font name='DVM'>swarm/planning/grid.py</font>, <font name='DVM'>swarm/planning/planner.py</font>",
        "Optimistic planning (treat unknown space as free, replan when walls are discovered) solved a maze "
        "nobody was told about. Tests proved the cost field finds paths exactly as short as A*, about 6× faster.")
    out += theory(
        "3.8 The assignment problem (Hungarian algorithm)",
        "Assign n followers to n formation slots minimising total travel distance. Solved exactly in O(n³).",
        ["min Σ<sub>i</sub> C<sub>i,σ(i)</sub> over permutations σ,  C<sub>ij</sub> = |p<sub>i</sub> − s<sub>j</sub>|"],
        "<font name='DVM'>swarm/formation/manager.py</font> (SciPy <i>linear_sum_assignment</i>)",
        "Re-solving the assignment after a shape change, or after drones fail, keeps the formation compact "
        "with minimal movement; a formation that loses its leader re-forms in seconds.")
    out += theory(
        "3.9 Particle Swarm Optimisation (PSO)",
        "A population of candidate solutions (particles) moves through the search space, each pulled toward "
        "its own best position and the swarm's best. Used twice: as a <i>physical</i> search (drones are the "
        "particles, seeking a victim's radio signal) and as an <i>offline optimiser</i> (a particle is a whole "
        "route plus a departure delay).",
        ["v ← w·v + c<sub>1</sub>r<sub>1</sub>(p<sub>best</sub> − x) + c<sub>2</sub>r<sub>2</sub>(g<sub>best</sub> − x);   x ← x + v",
         "victim search: w = 0.6, c<sub>1</sub> = 1.2, c<sub>2</sub> = 1.6, plus noise ∝ estimated range and restarts after 6 s stall",
         "routes: w decreases 0.9 → 0.4 (Shi & Eberhart, 1998); cost = length + threat + climb + turns + conflicts + delay"],
        "<font name='DVM'>swarm/mission/manager.py</font>, <font name='DVM'>swarm/terrain/route_planner.py</font>",
        "Pure PSO is unreliable with hard constraints (some routes clipped forbidden areas); seeding it with a "
        "graph-search route (hybrid) made it reliable. With only three particles and noisy signals, restarts "
        "and distance-scaled exploration were what made the victim search work.")
    out += theory(
        "3.10 Exploration, coverage and task allocation",
        "Frontier exploration sends robots to the boundary between known-free and unknown space (Yamauchi, "
        "1997). A Voronoi partition gives each robot the frontiers closer to it than to any teammate. Market or "
        "consensus auctions (CBBA) allocate tasks: the best bidders win.",
        ["drone k takes frontier f only if |f − p<sub>k</sub>| ≤ min<sub>j≠k</sub> |f − p<sub>j</sub>|  (believed positions from radio)",
         "auction: bid = distance to the best signal reading; keep seeking only while among the 3 lowest fresh bids"],
        "<font name='DVM'>swarm/mission/manager.py</font>",
        "Exploring the map is not the same as seeing every spot: frontier search found ~7 of 8 victims, the "
        "signal-guided PSO found the last ones, and switching the radio off dropped success from 8 to 3.3.")
    out += theory(
        "3.11 Reinforcement learning: PPO, GAE and centralised training",
        "Each drone's problem is a partially observable Markov decision process. Proximal Policy Optimisation "
        "improves a stochastic policy while clipping how far each update can move it. Generalised Advantage "
        "Estimation trades bias against variance. In multi-agent settings, a critic that sees team-wide "
        "information during training (centralised training, decentralised execution) stabilises learning.",
        ["L<sup>CLIP</sup>(θ) = E[ min( r<sub>t</sub>(θ)Â<sub>t</sub>,  clip(r<sub>t</sub>(θ), 1−ε, 1+ε)Â<sub>t</sub> ) ],  "
         "r<sub>t</sub> = π<sub>θ</sub>(a<sub>t</sub>|o<sub>t</sub>) / π<sub>old</sub>(a<sub>t</sub>|o<sub>t</sub>),  ε = 0.2",
         "Â<sub>t</sub> = Σ<sub>l</sub> (γλ)<sup>l</sup> δ<sub>t+l</sub>,   δ<sub>t</sub> = r<sub>t</sub> + γV(s<sub>t+1</sub>) − V(s<sub>t</sub>),  "
         "γ = 0.99, λ = 0.95"],
        "<font name='DVM'>swarm/rl/</font> (observation, environment, PPO, deployment as a controller)",
        "The details decide success: a shared gradient clip starved the actor; acceleration actions made every "
        "task a momentum-control problem; an unobservable penalty (arena walls the lidar could not see) made "
        "learning go backwards. Policies trained with 8 drones did not generalise to 20.")
    out += theory(
        "3.12 Fault tolerance and leader election",
        "A distributed system must keep working when members fail. Leader election chooses a new coordinator "
        "(e.g. the bully algorithm, Garcia-Molina 1982); graceful degradation means performance drops in "
        "proportion to the damage rather than collapsing.",
        ["new leader = min{ id : drone id is alive };  slots re-assigned for the survivors"],
        "<font name='DVM'>swarm/sim/engine.py</font> (fail), <font name='DVM'>swarm/formation/manager.py</font> (set_alive)",
        "With half the rescue swarm failing mid-mission, all 8 victims were still found. Noisy lidar (≥ 1 m) "
        "was more harmful than missing lidar beams: wrong data is worse than no data.")
    return out


def s4_tech():
    out = [PageBreak(), H("4. Technologies and tools", 0)]
    rows = [["Technology", "Used for", "Why it was chosen"],
            ["Python 3.12", "Everything", "Fast to iterate, the standard for robotics research prototypes and ML"],
            ["NumPy", "Vectorised state, physics, sensing, metrics",
             "Batch maths over all drones at once: 25× real time for 50 drones on a laptop CPU"],
            ["SciPy", "cKDTree neighbours; csgraph Dijkstra; linear_sum_assignment; ndimage distance transform",
             "Proven, compiled implementations of the core algorithms"],
            ["PyTorch (CPU)", "PPO actor and critic networks", "Standard deep-learning library; runs without a GPU"],
            ["Pygame-CE", "Real-time viewer, offscreen rendering for demos", "Lightweight 2D drawing; works headless"],
            ["Matplotlib", "Benchmark plots and figures", "Publication-quality charts"],
            ["Pillow", "Demo GIF encoding", "No external video tools needed"],
            ["PyYAML + dataclasses", "Typed configuration (configs/default.yaml)", "Every parameter in one validated place"],
            ["pytest", "189 unit and end-to-end tests", "Every phase verified before it was committed"],
            ["ReportLab", "This document", "PDF generated from code, so it stays in sync"],
            ["Git + GitHub", "Version control, development branch per phase", "Reviewable history, one commit per phase"]]
    out.append(table(rows, [3.2 * cm, 6.6 * cm, 7.2 * cm]))
    out.append(P("Hardware constraint: an Intel i5-8365U laptop (4 cores, 16 GB RAM, integrated graphics, no CUDA "
                 "GPU, limited disk). This ruled out Unity and GPU training and pushed the design toward efficient, "
                 "vectorised CPU code.", "small"))
    return out


def s5_architecture():
    out = [H("5. System architecture", 0)]
    out.append(P("Every simulation step (dt = 0.05 s) runs the same pipeline; each stage only uses information a "
                 "real drone could have."))
    rows = [["Stage", "What happens", "Module"],
            ["1. Sense", "Neighbours within 15 m (KD-tree); 36-beam lidar; nearest obstacle points", "sim/neighbors.py, agents/sensors.py"],
            ["2. Decide (mission)", "Rescue state machine, formation slots, terrain routes", "mission/, formation/, terrain/"],
            ["3. Plan", "Path on the drone's own map; steering waypoint by string pulling", "planning/"],
            ["4. Control", "Boids / formation / mission / learned policy → desired acceleration", "control/, rl/"],
            ["5. Safety", "ORCA turns it into a collision-free velocity", "control/orca.py"],
            ["6. Physics", "Integrate with speed/acceleration limits; walls; obstacle contact", "sim/dynamics.py, sim/world.py"],
            ["7. Map", "Fold the lidar scan into the drone's own occupancy grid", "mapping/occupancy.py"],
            ["8. Communicate", "Broadcast beacons, map cells, mission gossip; loss and latency", "comms/network.py"],
            ["9. Measure", "Collisions, clearance, coverage, arrivals, formation error, ...", "metrics.py"]]
    out.append(table(rows, [3.0 * cm, 8.8 * cm, 5.2 * cm]))
    out.append(Spacer(1, 6))
    out.append(P("<b>One controller interface.</b> Every behaviour implements "
                 "<font name='DVM'>compute(state, context) → acceleration</font>, so a hand-written controller, a "
                 "learned policy, or either wrapped in ORCA can be swapped with one setting and benchmarked under "
                 "identical conditions."))
    out += figure(FIG / "viewer_rescue.png", "The live viewer during a search-and-rescue mission (130 s): the team's "
                  "shared map (green), radio links, victims (✕), the base station (blue square) and live statistics.",
                  width=13 * cm)
    return out


def s6_results():
    out = [PageBreak(), H("6. Key results", 0)]
    out.append(P("All numbers come from the benchmark scripts in <font name='DVM'>experiments/</font>, repeated over "
                 "several random seeds. The full tables are in the project README."))
    out += figure(FIG / "avoidance.png", "Figure 1 - Only ORCA keeps every drone off every obstacle "
                  "(50 drones, 3 obstacle fields, 5 seeds).")
    out += figure(FIG / "comms_coverage.png", "Figure 2 - Multi-hop map sharing: the average drone knows 85% of the "
                  "arena after 30 s, against 17% without sharing; robust to heavy packet loss.", width=13 * cm)
    out += figure(FIG / "planning.png", "Figure 3 - In a maze, reactive flight never escapes; planning on each "
                  "drone's own map always does, and shared maps make it 29% faster.")
    out += figure(FIG / "rescue.png", "Figure 4 - Search and rescue: only the signal-guided PSO finds all 8 "
                  "victims in every run; without radio the swarm finds 3.3.", width=13 * cm)
    out += figure(FIG / "terrain.png", "Figure 5 - Planning in space and time: hybrid PSO routes have no planned "
                  "conflicts and 4–5× fewer close encounters than grid routes.")
    out += figure(FIG / "rl.png", "Figure 6 - Learned navigation is fast but unsafe alone; wrapped in ORCA it "
                  "is ~25% faster than the hand-written stack with almost no contact.")
    out += figure(FIG / "robustness.png", "Figure 7 - Graceful degradation under scaling, radio loss and "
                  "sensor noise.")
    return out


def s7_lessons():
    out = [PageBreak(), H("7. Lessons learned", 0)]
    out.append(P("Many of the most useful lessons came from bugs that tests and benchmarks exposed. These are "
                 "the ones worth remembering for any robotics or multi-agent project."))
    out.append(P("Research lessons", "h2"))
    out += bullets([
        "<b>Local rules really do produce global behaviour</b>, but only if safety has priority over everything "
        "else. Prioritised acceleration and ORCA turned an approximate flock into a provably collision-free one.",
        "<b>Communication multiplies capability.</b> Map sharing gave each drone 5× more knowledge; switching the "
        "radio off reduced rescue success by 60%. Robustness comes from redundancy (gossip, anti-entropy, relays).",
        "<b>Plan with what you know, assume optimistically, replan fast.</b> Optimistic planning on each drone's own "
        "map solved a maze none of them had seen.",
        "<b>Combine methods instead of choosing one.</b> PSO + graph search (hybrid), learned policy + ORCA, "
        "frontier exploration + signal search: every best result was a combination.",
        "<b>Report negative results.</b> Learned policies did not beat hand-written control at 20 drones. That is a "
        "real finding about data and compute budgets, not a failure to hide.",
        "<b>Wrong information is worse than missing information.</b> Losing 60% of lidar beams was harmless; 1 m of "
        "noise caused crashes."])
    out.append(P("Engineering lessons", "h2"))
    out += bullets([
        "<b>Measure, don't assume.</b> Profiling found that ray casting, graph construction and path extraction "
        "dominated; vectorisation made them 3–6× faster with identical results (checked against brute force).",
        "<b>Test the invariants, not just the outputs.</b> Tests caught a ray-casting bug that made drones “see” a "
        "wall inside a gap, path extraction that cut corners, and a route score that depended on its batch.",
        "<b>Metrics can lie.</b> A stuck detector fired ~200 false alarms per run; “left behind” counted a correct "
        "column as stranded. Every metric needs a sanity check against what actually happened.",
        "<b>In RL, an unobservable penalty makes learning go backwards</b>, and one shared gradient clip can silently "
        "stop the policy from learning. Verify the learner on a toy problem first.",
        "<b>Reproducibility needs separate random streams</b>: radio randomness must not change how drones fly, so "
        "experiments differ only in what they are meant to test."])
    return out


def s8_sim_to_real():
    out = [PageBreak(), H("8. From simulation to the real world", 0)]
    out.append(P("A simulation is valuable only if its ideas survive contact with reality. This section lists what "
                 "the simulation assumes, what real systems use instead, a step-by-step path to real drones, and the "
                 "real-world problems the project's methods address."))

    out.append(H("8.1 The reality gap", 1))
    rows = [["Simulation assumes", "Reality", "How to close the gap"],
            ["2D world, point-mass drones", "3D flight, quadrotor dynamics (thrust, attitude, drag, rotor lag)",
             "3D extension; low-level flight controller (PX4 / ArduPilot) tracks the velocity commands"],
            ["Perfect self-position", "GPS ±2–5 m (RTK ±2 cm); no GPS indoors; drift",
             "RTK-GPS outdoors; UWB, motion capture or visual-inertial odometry indoors; filter estimates"],
            ["Ideal lidar, Gaussian noise", "Reflections, dust, sunlight, weight and power limits",
             "Sensor fusion (lidar + depth camera); noise-aware mapping; the Phase 9 tests show where it breaks"],
            ["Simple radio (range, loss, latency)", "Interference, obstacles, bandwidth, security",
             "Mesh networking (Wi-Fi 802.11s, ESP-NOW, LoRa for long range); encryption; keep messages small"],
            ["No wind or weather", "Gusts, rain, temperature", "Robust control margins; ORCA with larger safety margins"],
            ["Linear battery model", "Voltage sag, temperature, payload effects", "Measured discharge curves; bigger reserve"],
            ["Instant computation", "Embedded CPUs, real-time limits", "Profile on target hardware (e.g. Jetson, Raspberry Pi); "
             "the vectorised design helps"],
            ["Learned policies trained in sim", "Sim-to-real gap for RL",
             "Domain randomisation, system identification, always keep a verified safety layer (ORCA)"]]
    out.append(table(rows, [4.0 * cm, 5.6 * cm, 7.4 * cm]))

    out.append(H("8.2 A step-by-step path to real drones", 1))
    steps = [["Step", "What to do", "Tools"],
             ["1. Higher-fidelity simulation", "Port the controllers to a 3D physics simulator with real quadrotor models",
              "Gazebo, PX4 SITL, gym-pybullet-drones, NVIDIA Isaac Sim"],
             ["2. Robot middleware", "Wrap each module (sensing, mapping, planning, ORCA, comms) as a ROS 2 node",
              "ROS 2, rclpy, Nav2 (for ideas)"],
             ["3. Flight-controller integration", "Send velocity setpoints in offboard mode; let the autopilot stabilise",
              "PX4 / ArduPilot, MAVLink, MAVSDK, MAVROS"],
             ["4. Small indoor swarm", "Test with tiny drones in a safe, netted room with external positioning",
              "Bitcraze Crazyflie + Crazyswarm2, Lighthouse / UWB / motion capture"],
             ["5. Real sensing and comms", "Add onboard range sensors and a mesh radio; tune the noise models from data",
              "Multi-ranger / ToF decks, small lidars, ESP-NOW / 802.11s mesh"],
             ["6. Outdoor field tests", "Few drones, geofenced, with pilot override, then scale up",
              "RTK-GPS, telemetry radios, QGroundControl"],
             ["7. Validation", "Repeat the project's benchmarks on hardware; compare to simulation",
              "Same metrics: collisions, clearance, coverage, arrival time"]]
    out.append(table(steps, [3.6 * cm, 7.0 * cm, 6.4 * cm]))

    out.append(H("8.3 Mapping each module to its real-world counterpart", 1))
    rows = [["Project module", "Real-world counterpart"],
            ["Boids + ORCA (control/)", "Onboard reactive collision avoidance; ORCA/RVO are used on real robots and "
             "drone swarms (e.g. RVO2 library)"],
            ["Lidar ray casting (agents/sensors.py)", "2D/3D lidar, time-of-flight sensors, depth cameras"],
            ["Occupancy grids (mapping/)", "OctoMap / 2D costmaps in ROS; SLAM systems for localisation + mapping"],
            ["Radio, gossip, beacons (comms/)", "Mesh networking, DDS (the ROS 2 transport), Remote ID broadcasts"],
            ["Planner (planning/)", "Global planners in navigation stacks; replanning on new map data"],
            ["Formations (formation/)", "Drone light shows, cooperative transport, convoy escort"],
            ["Rescue mission (mission/)", "Multi-UAV search missions with thermal cameras, phone-signal detection"],
            ["Route planning (terrain/)", "Pre-flight mission planning with airspace, terrain and no-fly constraints"],
            ["Learned policies (rl/)", "Learned controllers deployed with safety filters; sim-to-real with domain randomisation"],
            ["Fault injection (faults)", "Failure-mode testing, redundancy design, return-to-home and failsafes"]]
    out.append(table(rows, [5.4 * cm, 11.6 * cm]))

    out.append(H("8.4 Real-world applications", 1))
    rows = [["Application", "How the project's methods apply", "What else is needed"],
            ["Disaster search and rescue (earthquakes, floods, landslides)",
             "Frontier + Voronoi exploration, signal-guided PSO (phones), battery-aware returns, relaying reports "
             "to a base", "Thermal / RGB cameras, person detection, weather-proof drones, coordination with rescue teams"],
            ["Wildfire and environmental monitoring",
             "Coverage, threat-zone route planning (fire = threat), shared maps",
             "Thermal sensing, long-range radio, heat-tolerant hardware"],
            ["Precision agriculture", "Coverage planning, formations for spraying lanes, shared crop maps",
             "Multispectral cameras, payload management, regulation for spraying"],
            ["Infrastructure inspection (bridges, power lines, wind turbines)",
             "Path planning near structures, ORCA close to obstacles, multi-drone task allocation",
             "High-precision positioning, detailed 3D models"],
            ["Emergency communication relay", "Multi-hop radio, connectivity, formations that keep links alive",
             "Communications payloads, endurance (tethered or long-flight drones)"],
            ["Warehouse inventory and logistics", "Indoor navigation, collision avoidance in dense traffic, task allocation",
             "Indoor positioning, barcode / RFID scanning"],
            ["Drone light shows", "Formation shapes, Hungarian slot assignment, time-deconflicted trajectories",
             "Precise RTK positioning, choreography tools"],
            ["Delivery and urban air mobility", "Time-deconflicted route planning, threat (no-fly) avoidance",
             "Airspace integration (UTM), certification, Remote ID"]]
    out.append(table(rows, [4.0 * cm, 7.0 * cm, 6.0 * cm]))

    out.append(H("8.5 Safety, legal and ethical considerations", 1))
    out += bullets([
        "<b>Regulation.</b> Drone operations are governed by national aviation authorities, for example the Civil "
        "Aviation Authority of Sri Lanka, the FAA (USA, Part 107) and EASA (EU). Swarms usually need special "
        "permissions: beyond-visual-line-of-sight flight, multiple aircraft per pilot, and flight over people.",
        "<b>Safety by design.</b> Geofences, return-to-home on low battery or lost link, a human override, and a verified "
        "safety layer (here: ORCA) around any learned or experimental controller.",
        "<b>Privacy.</b> Cameras and phone-signal detection can collect personal data; search missions need clear "
        "purpose, data minimisation and deletion policies.",
        "<b>Dual use.</b> Swarm technology can be misused. The project focuses on humanitarian and civilian applications; "
        "responsible publication and adherence to law are part of the engineering.",
        "<b>Reliability honesty.</b> Report limitations openly (e.g. large formations, learned policies at scale) so "
        "nobody deploys a method beyond what has been demonstrated."])
    return out


def s9_future():
    out = [PageBreak(), H("9. Future research directions", 0)]
    out += bullets([
        "<b>3D swarm</b>: altitude as a real dimension, 3D ORCA, 3D occupancy (OctoMap-style) and terrain-following flight.",
        "<b>Incremental replanning</b>: full D* Lite instead of recomputing the cost field; any-angle paths (Theta*).",
        "<b>Distributed SLAM</b>: drones that estimate their own positions and merge maps without perfect localisation.",
        "<b>Better multi-agent learning</b>: graph neural network policies that work for any number of neighbours, longer "
        "training on a GPU (Colab / Kaggle), domain randomisation for sim-to-real.",
        "<b>Large formations</b>: hierarchical formations (squads with sub-leaders) for 50+ drones.",
        "<b>Consensus-based leader election and task allocation</b> that tolerate message loss and network partitions.",
        "<b>Charging logistics</b>: queueing at charging pads, battery swapping, mobile base stations.",
        "<b>Adversarial and security aspects</b>: spoofed messages, jamming, Byzantine-robust gossip.",
        "<b>Hardware validation</b> with a small Crazyflie swarm, repeating the project's benchmarks."])
    out.append(H("Glossary", 0))
    rows = [["Term", "Meaning"],
            ["Agent", "An autonomous decision maker; here, one drone"],
            ["Emergence", "Global behaviour arising from local interactions without central control"],
            ["Polarisation", "How aligned the swarm's headings are (0 random, 1 identical)"],
            ["Velocity obstacle", "Set of velocities that cause a collision within a time horizon"],
            ["Occupancy grid", "Map of cells each holding the probability of being occupied"],
            ["Log-odds", "log(p / (1 − p)); turns Bayesian updates into additions"],
            ["Frontier", "Boundary between known-free and unknown space"],
            ["Voronoi partition", "Division of space by which robot is nearest"],
            ["Admissible heuristic", "Never overestimates the remaining cost, so A* stays optimal"],
            ["Gossip / anti-entropy", "Spreading data by neighbour exchanges / periodically repairing missed updates"],
            ["PSO", "Particle Swarm Optimisation: population search guided by personal and global bests"],
            ["PPO / GAE", "Clipped policy-gradient RL algorithm / advantage estimator"],
            ["CTDE", "Centralised training, decentralised execution"],
            ["Sim-to-real gap", "Differences that make simulated results fail on real hardware"],
            ["Graceful degradation", "Performance that drops gradually, not catastrophically, under faults"]]
    out.append(table(rows, [4.0 * cm, 13.0 * cm]))
    return out


REFERENCES = [
    "C. W. Reynolds, “Flocks, herds and schools: A distributed behavioral model,” <i>Computer Graphics (SIGGRAPH)</i>, 21(4), pp. 25–34, 1987.",
    "C. W. Reynolds, “Steering behaviors for autonomous characters,” <i>Game Developers Conference</i>, 1999.",
    "T. Vicsek et al., “Novel type of phase transition in a system of self-driven particles,” <i>Physical Review Letters</i>, 75(6), p. 1226, 1995.",
    "R. Olfati-Saber, “Flocking for multi-agent dynamic systems: Algorithms and theory,” <i>IEEE Trans. Automatic Control</i>, 51(3), pp. 401–420, 2006.",
    "M. Brambilla, E. Ferrante, M. Birattari, M. Dorigo, “Swarm robotics: A review from the swarm engineering perspective,” <i>Swarm Intelligence</i>, 7(1), pp. 1–41, 2013.",
    "S.-J. Chung, A. A. Paranjape, P. Dames, S. Shen, V. Kumar, “A survey on aerial swarm robotics,” <i>IEEE Trans. Robotics</i>, 34(4), pp. 837–855, 2018.",
    "O. Khatib, “Real-time obstacle avoidance for manipulators and mobile robots,” <i>Int. J. Robotics Research</i>, 5(1), pp. 90–98, 1986.",
    "P. Fiorini, Z. Shiller, “Motion planning in dynamic environments using velocity obstacles,” <i>Int. J. Robotics Research</i>, 17(7), pp. 760–772, 1998.",
    "J. van den Berg, M. Lin, D. Manocha, “Reciprocal velocity obstacles for real-time multi-agent navigation,” <i>IEEE ICRA</i>, 2008.",
    "J. van den Berg, S. J. Guy, M. Lin, D. Manocha, “Reciprocal n-body collision avoidance,” <i>Robotics Research (ISRR)</i>, Springer STAR vol. 70, pp. 3–19, 2011.",
    "A. D. Ames et al., “Control barrier functions: Theory and applications,” <i>European Control Conference</i>, 2019.",
    "A. Elfes, “Using occupancy grids for mobile robot perception and navigation,” <i>Computer</i>, 22(6), pp. 46–57, 1989.",
    "S. Thrun, W. Burgard, D. Fox, <i>Probabilistic Robotics</i>, MIT Press, 2005.",
    "A. Demers et al., “Epidemic algorithms for replicated database maintenance,” <i>ACM PODC</i>, 1987.",
    "E. W. Dijkstra, “A note on two problems in connexion with graphs,” <i>Numerische Mathematik</i>, 1, pp. 269–271, 1959.",
    "P. E. Hart, N. J. Nilsson, B. Raphael, “A formal basis for the heuristic determination of minimum cost paths,” <i>IEEE Trans. Systems Science and Cybernetics</i>, 4(2), pp. 100–107, 1968.",
    "S. Koenig, M. Likhachev, “D* Lite,” <i>AAAI Conference on Artificial Intelligence</i>, 2002.",
    "A. Nash, K. Daniel, S. Koenig, A. Felner, “Theta*: Any-angle path planning on grids,” <i>AAAI</i>, 2007.",
    "J. P. Desai, J. P. Ostrowski, V. Kumar, “Modeling and control of formations of nonholonomic mobile robots,” <i>IEEE Trans. Robotics and Automation</i>, 17(6), pp. 905–908, 2001.",
    "W. Ren, R. W. Beard, <i>Distributed Consensus in Multi-vehicle Cooperative Control</i>, Springer, 2008.",
    "H. W. Kuhn, “The Hungarian method for the assignment problem,” <i>Naval Research Logistics Quarterly</i>, 2, pp. 83–97, 1955.",
    "J. Kennedy, R. Eberhart, “Particle swarm optimization,” <i>IEEE Int. Conf. Neural Networks</i>, 1995.",
    "Y. Shi, R. Eberhart, “A modified particle swarm optimizer,” <i>IEEE Int. Conf. Evolutionary Computation</i>, 1998.",
    "M. Erdmann, T. Lozano-Pérez, “On multiple moving objects,” <i>Algorithmica</i>, 2, pp. 477–521, 1987.",
    "D. H. Douglas, T. K. Peucker, “Algorithms for the reduction of the number of points required to represent a digitized line or its caricature,” <i>Cartographica</i>, 10(2), pp. 112–122, 1973.",
    "B. Yamauchi, “A frontier-based approach for autonomous exploration,” <i>IEEE CIRA</i>, 1997.",
    "J. Cortés, S. Martínez, T. Karataş, F. Bullo, “Coverage control for mobile sensing networks,” <i>IEEE Trans. Robotics and Automation</i>, 20(2), pp. 243–255, 2004.",
    "H.-L. Choi, L. Brunet, J. P. How, “Consensus-based decentralized auctions for robust task allocation,” <i>IEEE Trans. Robotics</i>, 25(4), pp. 912–926, 2009.",
    "H. Garcia-Molina, “Elections in a distributed computing system,” <i>IEEE Trans. Computers</i>, C-31(1), pp. 48–59, 1982.",
    "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, “Proximal policy optimization algorithms,” <i>arXiv:1707.06347</i>, 2017.",
    "J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, “High-dimensional continuous control using generalized advantage estimation,” <i>ICLR</i>, 2016.",
    "R. Lowe et al., “Multi-agent actor-critic for mixed cooperative-competitive environments,” <i>NeurIPS</i>, 2017.",
    "C. Yu et al., “The surprising effectiveness of PPO in cooperative multi-agent games,” <i>NeurIPS Datasets and Benchmarks</i>, 2022.",
    "S. Huang et al., “The 37 implementation details of proximal policy optimization,” <i>ICLR Blog Track</i>, 2022.",
    "J. Tobin et al., “Domain randomization for transferring deep neural networks from simulation to the real world,” <i>IEEE/RSJ IROS</i>, 2017.",
    "J. A. Preiss, W. Hönig, G. S. Sukhatme, N. Ayanian, “Crazyswarm: A large nano-quadcopter swarm,” <i>IEEE ICRA</i>, 2017.",
    "W. Hönig, J. A. Preiss, T. K. S. Kumar, G. S. Sukhatme, N. Ayanian, “Trajectory planning for quadrotor swarms,” <i>IEEE Trans. Robotics</i>, 34(4), pp. 856–869, 2018.",
    "J. Panerati et al., “Learning to fly: A gym environment with PyBullet physics for reinforcement learning of multi-agent quadcopter control,” <i>IEEE/RSJ IROS</i>, 2021.",
    "L. Meier, D. Honegger, M. Pollefeys, “PX4: A node-based multithreaded open source robotics framework for deeply embedded platforms,” <i>IEEE ICRA</i>, 2015.",
    "S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall, “Robot Operating System 2: Design, architecture, and uses in the wild,” <i>Science Robotics</i>, 7(66), 2022.",
    "S. Shah, D. Dey, C. Lovett, A. Kapoor, “AirSim: High-fidelity visual and physical simulation for autonomous vehicles,” <i>Field and Service Robotics</i>, Springer, 2018.",
]


def s10_refs():
    out = [PageBreak(), H("10. References", 0)]
    out.append(P("Key papers and books behind the methods used or studied in the project, grouped roughly in the "
                 "order the topics appear: swarms, avoidance, mapping, communication, planning, formations, "
                 "optimisation, missions, learning and real-world platforms."))
    for i, r in enumerate(REFERENCES, 1):
        out.append(Paragraph(f"[{i}] {r}", S["ref"]))
    out.append(Spacer(1, 10))
    out.append(P(f"Project source code, benchmarks and demos: <b>{REPO}</b>. This document is generated by "
                 "<font name='DVM'>python -m experiments.make_research_guide</font>.", "small"))
    return out


def build(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    story = cover() + toc_page()
    for part in (s1_overview, s2_research_map, s3_theories, s4_tech, s5_architecture, s6_results, s7_lessons,
                 s8_sim_to_real, s9_future, s10_refs):
        story += part()
    doc = Doc(path)
    doc.multiBuild(story)
    return path


if __name__ == "__main__":
    out = build()
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
