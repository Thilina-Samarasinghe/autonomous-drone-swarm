"""Radio network between drones.

Model
  * broadcast radio: a message reaches every drone within `range` metres
  * each delivery is lost independently with probability `packet_loss`
  * delivered messages arrive `latency` seconds later
  * drones broadcast every `broadcast_interval` seconds

Each message carries
  * a beacon table: where the sender believes every drone was, and when.
    Receivers keep whichever entry is newer, so positions spread hop by hop.
  * (optional) map cells: recently learned ones first, then a random sample
    of already-known ones, capped at `max_cells_per_message` (bandwidth limit)

With `relay` off, a message only describes the sender itself and receivers
never forward what they learned from others: communication is single-hop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from swarm.config import CommsConfig
from swarm.mapping.occupancy import OccupancyMaps

HEADER_BYTES = 16
BEACON_BYTES = 20      # x, y, vx, vy, timestamp as float32
CELL_BYTES = 3         # uint16 index + int8 value


@dataclass
class Broadcast:
    """One message from one sender, plus who will actually receive it."""
    sender: int
    deliver_step: int
    receivers: np.ndarray       # receivers whose copy was NOT lost
    table: np.ndarray           # (N, 3) x, y, time
    vtable: np.ndarray          # (N, 2) vx, vy at that time
    cells: np.ndarray           # (K,) flat cell indices
    values: np.ndarray          # (K,) log-odds


@dataclass
class CommsStats:
    broadcasts: int = 0
    attempted: int = 0          # directed deliveries attempted (one per in-range receiver)
    delivered: int = 0
    dropped: int = 0
    bytes_sent: int = 0
    cells_adopted: int = 0


@dataclass
class CommsNetwork:
    n: int
    cfg: CommsConfig
    dt: float
    rng: np.random.Generator
    queue: list[Broadcast] = field(init=False, default_factory=list)
    stats: CommsStats = field(init=False, default_factory=CommsStats)

    def __post_init__(self):
        self.links = np.empty((0, 2), dtype=int)          # (K, 2) undirected pairs in range
        self.recent_drops = np.empty((0, 2), dtype=int)   # (D, 2) sender, receiver at last broadcast
        self.table = np.full((self.n, self.n, 3), np.nan) # drone i's belief about drone j
        self.vtable = np.zeros((self.n, self.n, 2))       # ... and its velocity at that time
        # (sender, receiver) of every message delivered on the latest step; other
        # modules (e.g. the rescue mission) piggy-back their data on these links
        self.deliveries = np.empty((0, 2), dtype=int)
        self.interval_steps = max(1, round(self.cfg.broadcast_interval / self.dt))
        self.latency_steps = max(0, round(self.cfg.latency / self.dt))

    # ----------------------------------------------------------------- step
    def step(self, step_count: int, time: float, pos: np.ndarray,
             maps: OccupancyMaps | None, vel: np.ndarray | None = None) -> None:
        idx = np.arange(self.n)
        self.table[idx, idx, 0:2] = pos
        self.table[idx, idx, 2] = time
        if vel is not None:
            self.vtable[idx, idx] = vel
        if self.n >= 2:
            self.links = cKDTree(pos).query_pairs(self.cfg.range, output_type="ndarray")
        else:
            self.links = np.empty((0, 2), dtype=int)
        if step_count % self.interval_steps == 0:
            self._broadcast(step_count, maps)
        self._deliver(step_count, maps)

    def _broadcast(self, step_count: int, maps: OccupancyMaps | None) -> None:
        n = self.n
        if len(self.links):
            senders = np.concatenate([self.links[:, 0], self.links[:, 1]])
            receivers = np.concatenate([self.links[:, 1], self.links[:, 0]])
        else:
            senders = receivers = np.empty(0, dtype=int)
        lost = self.rng.random(len(senders)) < self.cfg.packet_loss
        self.recent_drops = np.stack([senders[lost], receivers[lost]], 1)
        self.stats.attempted += len(senders)
        self.stats.dropped += int(lost.sum())

        has_audience = np.bincount(senders, minlength=n) > 0
        deliver = step_count + self.latency_steps
        no_cells = np.empty(0, dtype=np.int64)
        no_vals = np.empty(0, dtype=np.float32)
        for s in np.flatnonzero(has_audience):
            if self.cfg.share_maps and maps is not None:
                cells, vals = maps.take_message(s, self.cfg.max_cells_per_message, self.rng,
                                                own_only=not self.cfg.relay)
            else:
                cells, vals = no_cells, no_vals
            table = self.table[s].copy() if self.cfg.relay else self._own_row_only(s)
            vtable = self.vtable[s].copy()
            self.stats.broadcasts += 1
            self.stats.bytes_sent += (HEADER_BYTES + CELL_BYTES * len(cells)
                                      + BEACON_BYTES * (n if self.cfg.relay else 1))
            ok = receivers[(senders == s) & ~lost]
            if len(ok):
                self.queue.append(Broadcast(int(s), deliver, ok, table, vtable, cells, vals))

    def _own_row_only(self, s: int) -> np.ndarray:
        t = np.full((self.n, 3), np.nan)
        t[s] = self.table[s, s]
        return t

    def _deliver(self, step_count: int, maps: OccupancyMaps | None) -> None:
        self.deliveries = np.empty((0, 2), dtype=int)
        if not self.queue:
            return
        ready = [b for b in self.queue if b.deliver_step <= step_count]
        if not ready:
            return
        self.queue = [b for b in self.queue if b.deliver_step > step_count]

        inbox: dict[int, list[Broadcast]] = {}
        pairs = []
        for b in ready:
            for r in b.receivers.tolist():
                if r < self.n and b.sender < self.n:
                    inbox.setdefault(r, []).append(b)
                    pairs.append((b.sender, r))
        if pairs:
            self.deliveries = np.array(pairs, dtype=int)

        for r, msgs in inbox.items():
            self.stats.delivered += len(msgs)
            # beacon tables: keep the newest entry per drone
            tables = np.stack([b.table for b in msgs] + [self.table[r]])      # (M+1, N, 3)
            times = np.nan_to_num(tables[:, :, 2], nan=-np.inf)
            best = times.argmax(axis=0)
            self.table[r] = tables[best, np.arange(self.n)]
            vtables = np.stack([b.vtable for b in msgs] + [self.vtable[r]])
            self.vtable[r] = vtables[best, np.arange(self.n)]
            if maps is not None:
                cells = np.concatenate([b.cells for b in msgs])
                if len(cells):
                    vals = np.concatenate([b.values for b in msgs])
                    self.stats.cells_adopted += maps.merge(r, cells, vals, self.cfg.relay)

    # -------------------------------------------------------------- queries
    def groups(self) -> int:
        """Connected components of the communication graph."""
        if self.n == 0:
            return 0
        if len(self.links) == 0:
            return self.n
        g = coo_matrix((np.ones(len(self.links)), (self.links[:, 0], self.links[:, 1])),
                       shape=(self.n, self.n))
        return int(connected_components(g, directed=False)[0])

    def knowledge(self, time: float) -> tuple[float, float]:
        """(fraction of teammates each drone knows about, mean age of that info in s)."""
        if self.n < 2:
            return 1.0, 0.0
        off = ~np.eye(self.n, dtype=bool)
        t = self.table[:, :, 2][off]
        known = ~np.isnan(t)
        age = float((time - t[known]).mean()) if known.any() else float("nan")
        return float(known.mean()), age
