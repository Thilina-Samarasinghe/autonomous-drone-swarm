"""Formation shapes as slot offsets in the leader's frame.

Frame: +x = the leader's heading (forward), +y = to the leader's left.
Slot 0 is always the leader at (0, 0). Every generator returns exactly `n`
slots, nearest-to-leader first, and scales to large swarms by adding
ranks / layers / rings instead of growing one enormous line.
"""

from __future__ import annotations

import math

import numpy as np

SHAPES = ("line", "v", "circle", "box", "column")
MAX_PER_RANK = 11          # line: drones per rank before starting a new rank behind
V_ARM = 5                  # v: drones per arm before nesting another V behind


def _alternate(k: int) -> int:
    """0 -> 0, 1 -> +1, 2 -> -1, 3 -> +2, 4 -> -2, ... (fill outward from the centre)."""
    return (k + 1) // 2 * (1 if k % 2 else -1)


def slot_offsets(shape: str, n: int, spacing: float) -> np.ndarray:
    if shape not in SHAPES:
        raise ValueError(f"Unknown formation shape '{shape}'. Choose from {SHAPES}")
    if n <= 0:
        return np.zeros((0, 2))
    s = spacing
    pts: list[tuple[float, float]] = [(0.0, 0.0)]

    if shape == "line":
        # ranks abreast; the leader in the middle of the front rank
        pts = [(-(i // MAX_PER_RANK) * s, _alternate(i % MAX_PER_RANK) * s) for i in range(n)]

    elif shape == "column":
        pts += [(-i * s, 0.0) for i in range(1, n)]

    elif shape == "v":
        # chevrons: arms at +/-45 degrees behind the apex; nest a new V behind when full
        d = s / math.sqrt(2) * 1.15
        layer = 0
        while len(pts) < n:
            apex = -layer * 2.2 * s
            if layer > 0:
                pts.append((apex, 0.0))
            for j in range(1, V_ARM + 1):
                for side in (1, -1):
                    if len(pts) < n:
                        pts.append((apex - j * d, side * j * d))
            layer += 1

    elif shape == "circle":
        # the leader at the centre, rings around it with ~spacing between neighbours
        ring = 1
        while len(pts) < n:
            r = ring * s * 1.2
            cap = max(6, int(2 * math.pi * r / s))
            take = min(cap, n - len(pts))
            for j in range(take):
                a = 2 * math.pi * j / take
                pts.append((r * math.cos(a), r * math.sin(a)))
            ring += 1

    elif shape == "box":
        # a square grid with the leader at the middle of the front edge
        side = max(1, math.ceil(math.sqrt(n)))
        cells = []
        for row in range(side + 1):
            for col in range(side):
                cells.append((-row * s, _alternate(col) * s))
        pts = cells[:n]

    return np.array(pts[:n], dtype=float)


def width(offsets: np.ndarray) -> float:
    """Lateral extent of a formation (m), used to decide whether it fits a gap."""
    if len(offsets) == 0:
        return 0.0
    return float(offsets[:, 1].max() - offsets[:, 1].min())


def to_world(offsets: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return origin[None] + offsets @ rot.T
